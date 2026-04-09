#!/usr/bin/env python3
"""
TriAttention KV Cache Compression for NF4 Models.

Full implementation of TriAttention (arxiv:2604.04921) for NF4-quantized
HuggingFace transformer models using bitsandbytes.

Key features:
  - Pre-RoPE Q/K center calibration via attention layer hooks
  - Trigonometric series scoring (S_trig) with post-RoPE key trick
  - Norm-based scoring (S_norm) with adaptive MRL concentration weighting
  - GQA support with normalize-then-aggregate (z-score + max)
  - Future offset averaging with geometric spacing {1,2,4,...,2^16}
  - Window-based pruning every β=128 tokens
  - Attention sink protection + recent window protection
  - Custom generation loop with explicit position tracking

  - tested with the models: 
  -- https://huggingface.co/g023/Qwopus3.5-9B-v3-NF4
  -- https://huggingface.co/g023/Qwen3-1.77B-g023-NF4

Based on: "TriAttention: Efficient Long Reasoning with Trigonometric KV
Compression" (Mao et al., 2026, arxiv:2604.04921)

Author: g023 (github.com/g023)
License: MIT
"""

import argparse
import math
import sys
import time
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# =============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =============================================================================
# Model loading
MODEL_PATH = "./Qwen3-NF4"
DEVICE_MAP = "auto"

# Sampling parameters
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 40
MIN_P = 0.05
REPETITION_PENALTY = 1.1
SEED = 42

# TriAttention pruning parameters
KV_BUDGET = 2048       # Maximum tokens to keep in KV cache
WINDOW_SIZE = 128      # Prune every N tokens (paper: β=128)
RECENT_TOKENS = 128    # Number of most recent tokens to always keep
SINK_TOKENS = 4        # Number of initial tokens to always keep (attention sinks)

# Future offset set D = {2^0, 2^1, ..., 2^16} for score averaging (paper Section 4.2)
FUTURE_OFFSETS = [2**i for i in range(17)]  # 17 offsets: 1, 2, 4, ..., 65536

# Generation
MAX_TOKENS = 8192      # Maximum tokens to generate

# Calibration
MIN_CALIBRATION_TOKENS = 32  # Minimum tokens for calibration

# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ARCHITECTURE DETECTION
# =============================================================================

def get_model_arch_info(model):
    """
    Detect model architecture specifics for TriAttention.

    Returns dict with:
      - full_attn_layer_indices: list of int (layers with KV cache)
      - partial_rotary_factor: float (1.0 for full RoPE)
      - rotary_dim: int (number of dims that get RoPE)
      - half_dim: int (rotary_dim // 2, for complex representation)
      - has_gated_q: bool (Qwen3.5 gates Q output)
      - is_hybrid: bool (True if model mixes linear + full attention)
    """
    config = model.config
    head_dim = getattr(config, "head_dim", None) or \
        config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers

    # Detect full_attention layer indices
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        full_attn_indices = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        is_hybrid = any(t != "full_attention" for t in layer_types)
    else:
        full_attn_indices = list(range(num_layers))
        is_hybrid = False

    # Partial rotary factor
    prf = getattr(config, "partial_rotary_factor", None)
    if prf is None and hasattr(config, "rope_parameters"):
        prf = config.rope_parameters.get("partial_rotary_factor", None)
    if prf is None:
        prf = 1.0

    rotary_dim = int(head_dim * prf)
    half_dim = rotary_dim // 2

    # Gated Q detection
    has_gated_q = getattr(config, "attn_output_gate", False)

    info = {
        "full_attn_layer_indices": full_attn_indices,
        "partial_rotary_factor": prf,
        "rotary_dim": rotary_dim,
        "half_dim": half_dim,
        "has_gated_q": has_gated_q,
        "is_hybrid": is_hybrid,
        "head_dim": head_dim,
    }
    logger.info(f"Architecture: {config.architectures[0] if hasattr(config, 'architectures') else 'unknown'}, "
                f"full_attn_layers={len(full_attn_indices)}/{num_layers}, "
                f"rotary_dim={rotary_dim}/{head_dim}, gated_q={has_gated_q}")
    return info


# =============================================================================
# ROPE FREQUENCY UTILITIES
# =============================================================================

def get_rope_frequencies(model, arch_info=None):
    """
    Extract RoPE inverse frequencies from the model's rotary embedding.

    For models with partial_rotary_factor < 1.0 (e.g. Qwen3.5), only
    computes frequencies for the rotary portion of head_dim.

    Returns tensor of shape [rotary_dim/2] with ω_f = θ^(-2f/d).
    """
    config = model.config
    rope_theta = config.rope_parameters.get("rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    # Determine rotary dimension
    if arch_info:
        rotary_dim = arch_info["rotary_dim"]
    else:
        prf = getattr(config, "partial_rotary_factor", None)
        if prf is None and hasattr(config, "rope_parameters"):
            prf = config.rope_parameters.get("partial_rotary_factor", None)
        rotary_dim = int(head_dim * (prf or 1.0))

    # ω_f = θ^(-2f/d) = 1 / θ^(2f/d)
    # Note: freq base uses rotary_dim (not full head_dim)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    return inv_freq  # shape: [rotary_dim/2]


# =============================================================================
# CALIBRATION: Pre-RoPE Q/K Statistics
# =============================================================================

class TriAttentionCalibrator:
    """
    Collects pre-RoPE Q/K statistics for TriAttention scoring.

    Hooks into attention layers to capture Q/K vectors after QKNorm
    but before RoPE rotation. Accumulates running statistics per head
    per layer per frequency band.

    Statistics collected:
      - q_center[layer][head]: E[q_f] as complex tensor [d/2]
      - q_norm_mean[layer][head]: E[||q_f||] tensor [d/2]
      - mrl[layer][head]: Mean Resultant Length R_f tensor [d/2]
    """

    def __init__(self, model, arch_info=None):
        self.model = model
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_q_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", None) or \
            self.config.hidden_size // self.config.num_attention_heads

        # Architecture-specific settings
        if arch_info:
            self.full_attn_indices = arch_info["full_attn_layer_indices"]
            self.half_dim = arch_info["half_dim"]
            self.rotary_dim = arch_info["rotary_dim"]
            self.has_gated_q = arch_info["has_gated_q"]
        else:
            self.full_attn_indices = list(range(self.num_layers))
            self.half_dim = self.head_dim // 2
            self.rotary_dim = self.head_dim
            self.has_gated_q = False

        # Running statistics accumulators
        # Per layer, per head: complex sum and norm sum
        self.q_complex_sum = {}  # [layer] -> [num_q_heads, half_dim] complex
        self.q_norm_sum = {}     # [layer] -> [num_q_heads, half_dim] float
        self.q_count = {}        # [layer] -> int

        self.hooks = []
        self._captured_pre_rope_q = {}

    def _install_hooks(self):
        """Install forward hooks to capture pre-RoPE Q/K from attention layers."""
        self._remove_hooks()

        # Access model layers
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            layers = self.model.layers

        has_gated_q = self.has_gated_q
        rotary_dim = self.rotary_dim

        for layer_idx in self.full_attn_indices:
            layer = layers[layer_idx]
            attn = layer.self_attn

            # We monkey-patch the forward to capture pre-RoPE Q
            # This is the cleanest way to get Q after q_norm but before RoPE
            original_forward = attn.forward.__func__ if hasattr(attn.forward, '__func__') else attn.forward

            def make_hook(l_idx, orig_fwd, attn_module, gated_q, rot_dim):
                def hooked_forward(self_attn, hidden_states, position_embeddings,
                                   attention_mask=None, past_key_values=None,
                                   cache_position=None, **kwargs):
                    # Capture pre-RoPE Q/K
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self_attn.head_dim)

                    if gated_q:
                        # Qwen3.5: q_proj outputs 2x width, split into query + gate
                        q_proj_out = self_attn.q_proj(hidden_states)
                        q_proj_view = q_proj_out.view(*input_shape, -1, self_attn.head_dim * 2)
                        query_states, _gate = torch.chunk(q_proj_view, 2, dim=-1)
                        pre_rope_q = self_attn.q_norm(
                            query_states.view(hidden_shape)
                        ).transpose(1, 2)
                    else:
                        # Qwen3: standard q_proj
                        pre_rope_q = self_attn.q_norm(
                            self_attn.q_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)  # [batch, num_q_heads, seq, head_dim]

                    # Only use rotary portion for calibration stats
                    if rot_dim < self_attn.head_dim:
                        pre_rope_q = pre_rope_q[..., :rot_dim]

                    # Store for calibration
                    calibrator_ref = self_attn._calibrator_ref
                    if calibrator_ref is not None:
                        calibrator_ref._accumulate_q_stats(l_idx, pre_rope_q)

                    # Call original forward
                    return orig_fwd(self_attn, hidden_states,
                                    position_embeddings=position_embeddings,
                                    attention_mask=attention_mask,
                                    past_key_values=past_key_values,
                                    cache_position=cache_position, **kwargs)
                return hooked_forward

            # Store reference to calibrator on the attention module
            attn._calibrator_ref = self

            # Bind the hooked forward
            import types
            attn.forward = types.MethodType(
                make_hook(layer_idx, original_forward, attn, has_gated_q, rotary_dim), attn
            )
            self.hooks.append((attn, original_forward))

    def _remove_hooks(self):
        """Remove hooks and restore original forward methods."""
        import types
        for attn, original_forward in self.hooks:
            attn.forward = types.MethodType(original_forward, attn)
            attn._calibrator_ref = None
        self.hooks = []

    def _accumulate_q_stats(self, layer_idx, pre_rope_q):
        """
        Accumulate Q statistics from a batch of pre-RoPE Q vectors.

        Args:
            layer_idx: Layer index
            pre_rope_q: [batch, num_q_heads, seq_len, head_dim]
        """
        with torch.no_grad():
            # Convert to complex representation
            # z_f = q[..., f] + i * q[..., f + d/2]
            q_real = pre_rope_q[..., :self.half_dim].float()  # [B, H, S, D/2]
            q_imag = pre_rope_q[..., self.half_dim:].float()  # [B, H, S, D/2]
            q_complex = torch.complex(q_real, q_imag)  # [B, H, S, D/2]

            # Norms per frequency band
            q_norms = q_complex.abs()  # [B, H, S, D/2]

            # Sum over batch and sequence dimensions
            # q_complex: [B, H, S, D/2] -> sum to [H, D/2]
            complex_sum = q_complex.sum(dim=(0, 2))  # [H, D/2]
            norm_sum = q_norms.sum(dim=(0, 2))  # [H, D/2]
            count = pre_rope_q.shape[0] * pre_rope_q.shape[2]  # batch * seq

            if layer_idx not in self.q_complex_sum:
                self.q_complex_sum[layer_idx] = complex_sum
                self.q_norm_sum[layer_idx] = norm_sum
                self.q_count[layer_idx] = count
            else:
                self.q_complex_sum[layer_idx] += complex_sum
                self.q_norm_sum[layer_idx] += norm_sum
                self.q_count[layer_idx] += count

    def calibrate(self, input_ids):
        """
        Run calibration on the given input tokens.

        Args:
            input_ids: Token IDs tensor [1, seq_len]

        Returns:
            dict with calibration data per layer:
                q_center: E[q_f] complex tensor [num_q_heads, half_dim]
                q_norm_mean: E[||q_f||] tensor [num_q_heads, half_dim]
                mrl: R_f tensor [num_q_heads, half_dim]
        """
        self._install_hooks()

        logger.info(f"Calibrating on {input_ids.shape[1]} tokens...")

        with torch.no_grad():
            # Forward pass to collect statistics
            self.model(input_ids, use_cache=False)

        self._remove_hooks()

        # Compute final statistics
        calibration_data = {}
        for layer_idx in self.full_attn_indices:
            if layer_idx not in self.q_complex_sum:
                logger.warning(f"No calibration data for layer {layer_idx}")
                continue

            count = self.q_count[layer_idx]
            q_center = self.q_complex_sum[layer_idx] / count  # E[q_f], [H, D/2]
            q_norm_mean = self.q_norm_sum[layer_idx] / count  # E[||q_f||], [H, D/2]

            # Mean Resultant Length: R_f = ||E[q_f]|| / E[||q_f||]
            q_center_norm = q_center.abs()  # ||E[q_f]||, [H, D/2]
            mrl = q_center_norm / (q_norm_mean + 1e-8)  # R_f, [H, D/2]
            mrl = mrl.clamp(0.0, 1.0)

            calibration_data[layer_idx] = {
                'q_center': q_center,        # complex [num_q_heads, half_dim]
                'q_center_norm': q_center_norm,  # float [num_q_heads, half_dim]
                'q_center_phase': q_center.angle(),  # float [num_q_heads, half_dim]
                'q_norm_mean': q_norm_mean,  # float [num_q_heads, half_dim]
                'mrl': mrl,                  # float [num_q_heads, half_dim]
            }

        # Clear accumulators
        self.q_complex_sum.clear()
        self.q_norm_sum.clear()
        self.q_count.clear()

        logger.info(f"Calibration complete for {len(calibration_data)} layers")
        avg_mrl = torch.stack([d['mrl'].mean() for d in calibration_data.values()]).mean()
        logger.info(f"Average MRL across all heads: {avg_mrl:.4f}")

        return calibration_data


# =============================================================================
# SCORING ENGINE
# =============================================================================

class TriAttentionScorer:
    """
    GPU-vectorized TriAttention scoring engine.

    Computes importance scores for cached keys using:
      - S_trig: Trigonometric series score (Eq 6)
      - S_norm: Norm-based score with MRL weighting (Eq 8)
      - Future offset averaging (Eq 11)
      - GQA normalize-then-aggregate (Eq 12-13)
    """

    def __init__(self, calibration_data, rope_inv_freq, num_q_heads, num_kv_heads,
                 future_offsets=None, rotary_dim=None):
        self.calibration_data = calibration_data
        self.rope_inv_freq = rope_inv_freq  # [D/2]
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.gqa_ratio = num_q_heads // num_kv_heads  # G
        self.rotary_dim = rotary_dim  # If set, only use first rotary_dim of key for scoring

        if future_offsets is None:
            future_offsets = FUTURE_OFFSETS
        self.future_offsets = torch.tensor(future_offsets, dtype=torch.float32)

    def score_keys(self, layer_idx, cached_keys, current_pos):
        """
        Score all cached keys for a given layer (fully vectorized).

        Args:
            layer_idx: Layer index
            cached_keys: Post-RoPE cached keys [1, num_kv_heads, seq_len, head_dim]
            current_pos: Current query position (int)

        Returns:
            Tensor of final scores [seq_len] (higher = more important)
        """
        cal = self.calibration_data[layer_idx]
        device = cached_keys.device

        seq_len = cached_keys.shape[2]
        G = self.gqa_ratio
        KV = self.num_kv_heads

        # For partial rotary models, only use the rotary portion of keys
        if self.rotary_dim is not None and self.rotary_dim < cached_keys.shape[3]:
            keys_for_scoring = cached_keys[0, :, :, :self.rotary_dim]  # [KV, S, rotary_dim]
        else:
            keys_for_scoring = cached_keys[0]  # [KV, S, head_dim]

        half_dim = keys_for_scoring.shape[2] // 2

        # Convert cached keys to complex: k̃_f = k[f] + i·k[f+d/2]
        k_real = keys_for_scoring[:, :, :half_dim].float()  # [KV, S, D/2]
        k_imag = keys_for_scoring[:, :, half_dim:].float()
        k_complex = torch.complex(k_real, k_imag)
        k_norms = k_complex.abs()    # [KV, S, D/2]
        k_phases = k_complex.angle()  # [KV, S, D/2]

        # Calibration data → device
        q_center_norm = cal['q_center_norm'].to(device)  # [Q, D/2]
        q_center_phase = cal['q_center_phase'].to(device)
        q_norm_mean = cal['q_norm_mean'].to(device)
        mrl = cal['mrl'].to(device)

        inv_freq = self.rope_inv_freq.to(device)  # [D/2]
        offsets = self.future_offsets.to(device)  # [|D|]

        # Reshape Q stats for GQA: [KV, G, D/2]
        q_cn = q_center_norm.view(KV, G, half_dim)
        q_cp = q_center_phase.view(KV, G, half_dim)
        q_nm = q_norm_mean.view(KV, G, half_dim)
        q_mrl = mrl.view(KV, G, half_dim)

        # ── S_trig: fully vectorized across offsets, keys, freq bands ──
        # base_phase = ω_f * (p_q + δ): [|D|, D/2]
        base_phase = inv_freq[None, :] * (current_pos + offsets[:, None])

        # Full phase: [|D|, KV, G, S, D/2]
        # = base_phase[|D|,1,1,1,D/2] + q_phase[1,KV,G,1,D/2] - k_phase[1,KV,1,S,D/2]
        phases = (base_phase[:, None, None, None, :]
                  + q_cp[None, :, :, None, :]
                  - k_phases[None, :, None, :, :])

        # Amplitude: [1, KV, G, S, D/2]
        amp = q_cn[None, :, :, None, :] * k_norms[None, :, None, :, :]

        # S_trig: sum over freq, mean over offsets → [KV, G, S]
        s_trig = (amp * phases.cos()).sum(dim=-1).mean(dim=0)

        # ── S_norm: [KV, G, S] ──
        norm_weight = (1.0 - q_mrl)  # [KV, G, D/2]
        s_norm = (norm_weight[:, :, None, :] * q_nm[:, :, None, :] * k_norms[:, None, :, :]).sum(dim=-1)
        # [KV, G, S]

        # Combined: [KV, G, S]
        s_combined = s_trig + s_norm

        # ── GQA: z-normalize per Q head, then max across G ──
        mu = s_combined.mean(dim=-1, keepdim=True)  # [KV, G, 1]
        sigma = s_combined.std(dim=-1, keepdim=True) + 1e-8
        z_scores = (s_combined - mu) / sigma  # [KV, G, S]

        # Max across Q heads per KV head: [KV, S]
        kv_scores = z_scores.max(dim=1).values

        # Mean across KV heads: [S]
        final_scores = kv_scores.mean(dim=0)

        return final_scores


# =============================================================================
# KV CACHE PRUNER
# =============================================================================

class TriAttentionPruner:
    """
    TriAttention KV cache pruner.

    Scores cached keys and evicts the lowest-scoring ones to meet the budget.
    Protects attention sinks (first N tokens) and recent window (last M tokens).
    Triggers pruning every window_size generation steps when cache exceeds budget.
    """

    def __init__(self, scorer, budget=KV_BUDGET, window_size=WINDOW_SIZE,
                 recent_tokens=RECENT_TOKENS, sink_tokens=SINK_TOKENS,
                 full_attn_layer_indices=None):
        self.scorer = scorer
        self.budget = budget
        self.window_size = window_size
        self.recent_tokens = recent_tokens
        self.sink_tokens = sink_tokens
        self.full_attn_layer_indices = full_attn_layer_indices  # If None, all layers

        # State
        self.gen_step = 0
        self.total_pruned = 0
        self.prune_count = 0

    def _get_cache_size(self, past_key_values):
        """Get KV cache size from a full_attention layer."""
        if self.full_attn_layer_indices:
            # Use first full_attention layer for seq_length
            idx = self.full_attn_layer_indices[0]
            if idx < len(past_key_values.layers):
                return past_key_values.get_seq_length(idx)
        return past_key_values.get_seq_length()

    def should_prune(self, cache_size):
        """Check if pruning should trigger."""
        return (cache_size > self.budget and
                self.gen_step > 0 and
                self.gen_step % self.window_size == 0)

    def prune(self, past_key_values, current_pos):
        """
        Score all cached keys and evict lowest-scoring ones.

        Args:
            past_key_values: DynamicCache object
            current_pos: Current query position

        Returns:
            Modified past_key_values with evicted entries removed
        """
        # Get cache size from a full_attention layer
        cache_size = self._get_cache_size(past_key_values)
        if cache_size <= self.budget:
            return past_key_values

        # Determine which layer indices to score/prune
        if self.full_attn_layer_indices:
            scoreable_layers = [i for i in self.full_attn_layer_indices
                                if i in self.scorer.calibration_data]
        else:
            scoreable_layers = list(range(len(past_key_values.layers)))

        # Determine protected indices
        n_sink = min(self.sink_tokens, cache_size)
        n_recent = min(self.recent_tokens, cache_size)

        # Score keys across sampled full_attention layers
        first_full_attn = scoreable_layers[0] if scoreable_layers else 0
        device = past_key_values.layers[first_full_attn].keys.device
        all_scores = torch.zeros(cache_size, device=device, dtype=torch.float32)

        # Sample ~8 layers from the full_attention layers
        n_scoreable = len(scoreable_layers)
        sample_stride = max(1, n_scoreable // 8)

        scored_layers = 0
        for i in range(0, n_scoreable, sample_stride):
            layer_idx = scoreable_layers[i]
            cached_keys = past_key_values.layers[layer_idx].keys
            layer_scores = self.scorer.score_keys(layer_idx, cached_keys, current_pos)
            all_scores += layer_scores
            scored_layers += 1

        # Build keep mask
        keep_mask = torch.zeros(cache_size, dtype=torch.bool, device=device)

        # Protect sinks
        keep_mask[:n_sink] = True

        # Protect recent
        keep_mask[-n_recent:] = True

        n_protected = keep_mask.sum().item()
        middle_budget = max(0, self.budget - n_protected)

        # Score middle tokens (not protected)
        middle_mask = ~keep_mask
        if middle_mask.any() and middle_budget > 0:
            middle_scores = all_scores.clone()
            middle_scores[keep_mask] = float('-inf')  # Exclude protected from ranking

            # Get top-K middle indices
            _, top_indices = middle_scores.topk(
                min(middle_budget, middle_mask.sum().item())
            )
            keep_mask[top_indices] = True

        # Apply pruning only to layers that have KV caches
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        n_evicted = cache_size - keep_indices.shape[0]

        if n_evicted <= 0:
            return past_key_values

        for layer_idx in range(len(past_key_values.layers)):
            layer = past_key_values.layers[layer_idx]
            # Only prune layers that have KV caches (full_attention layers)
            if not hasattr(layer, 'keys') or layer.keys is None or layer.keys.numel() == 0:
                continue
            # layer.keys: [batch, heads, seq, head_dim]
            layer.keys = layer.keys[:, :, keep_indices, :].contiguous()
            layer.values = layer.values[:, :, keep_indices, :].contiguous()

        self.total_pruned += n_evicted
        self.prune_count += 1

        new_size = self._get_cache_size(past_key_values)
        logger.debug(f"Pruned {n_evicted} tokens: {cache_size} → {new_size} "
                     f"(budget={self.budget})")

        return past_key_values


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_triattention(model, tokenizer, input_ids, pruner,
                               max_new_tokens=MAX_TOKENS,
                               temperature=TEMPERATURE, top_p=TOP_P,
                               top_k=TOP_K, min_p=MIN_P,
                               repetition_penalty=REPETITION_PENALTY,
                               seed=SEED, streaming=True, quiet=False):
    """
    Generate tokens with TriAttention KV cache pruning.

    Uses a custom generation loop with explicit position tracking
    to handle KV cache pruning correctly.

    Returns:
        (generated_text, stats_dict)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = input_ids.device
    n_prompt = input_ids.shape[1]

    # Initialize KV cache — let model create its own (hybrid-compatible)
    past_key_values = None

    # Process prompt
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

    next_token_logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    current_pos = n_prompt  # Next token will be at this position

    generated_tokens = []
    generated_text_parts = []
    start_time = time.perf_counter()

    eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        # Apply sampling
        logits = next_token_logits.float()

        # Repetition penalty
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            prev_tokens = torch.tensor(generated_tokens[-256:], device=device)
            for tok in prev_tokens.unique():
                if logits[0, tok] > 0:
                    logits[0, tok] /= repetition_penalty
                else:
                    logits[0, tok] *= repetition_penalty

        # Temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            text = tokenizer.decode([next_token.item()], skip_special_tokens=False)
            generated_text_parts.append(text)
            if not quiet and streaming:
                print(text, end='', flush=True)
            if next_token.item() == eos_token_id:
                break

            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=torch.tensor([current_pos], device=device),
                    position_ids=torch.tensor([[current_pos]], device=device),
                )
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            current_pos += 1
            pruner.gen_step += 1

            cache_size = pruner._get_cache_size(past_key_values)
            if pruner.should_prune(cache_size):
                past_key_values = pruner.prune(past_key_values, current_pos)

            continue

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Min-p filtering
        if min_p > 0:
            probs = F.softmax(logits, dim=-1)
            max_prob = probs.max(dim=-1, keepdim=True).values
            min_threshold = max_prob * min_p
            logits[probs < min_threshold] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        generated_tokens.append(token_id)

        # Decode and output
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        generated_text_parts.append(text)
        if not quiet and streaming:
            print(text, end='', flush=True)

        if token_id == eos_token_id:
            break

        # Forward pass for next token
        with torch.no_grad():
            outputs = model(
                input_ids=next_token.view(1, 1),
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=torch.tensor([current_pos], device=device),
                position_ids=torch.tensor([[current_pos]], device=device),
            )

        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        current_pos += 1
        pruner.gen_step += 1

        # Check for pruning
        cache_size = pruner._get_cache_size(past_key_values)
        if pruner.should_prune(cache_size):
            past_key_values = pruner.prune(past_key_values, current_pos)

    elapsed = time.perf_counter() - start_time
    n_gen = len(generated_tokens)

    # CUDA memory tracking
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    stats = {
        'prompt_tokens': n_prompt,
        'generated_tokens': n_gen,
        'total_time': elapsed,
        'tokens_per_sec': n_gen / elapsed if elapsed > 0 else 0,
        'prune_events': pruner.prune_count,
        'total_pruned': pruner.total_pruned,
        'final_cache_size': pruner._get_cache_size(past_key_values),
        'peak_vram_mb': peak_mem_mb,
    }

    full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return full_text, stats


# =============================================================================
# MAIN
# =============================================================================

def load_model(model_path, device_map=DEVICE_MAP):
    """Load NF4 model and tokenizer."""
    logger.info(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # device_map=device_map,
        # dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Model loaded.")
    return model, tokenizer


def prepare_prompt(tokenizer, prompt, system_prompt=None):
    """Format prompt using chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
    except (TypeError, Exception) as e:
        if 'enable_thinking' in str(e) or 'UndefinedError' in type(e).__name__ or isinstance(e, TypeError):
            # Qwen3.5 and some models don't support enable_thinking
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            raise
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    return input_ids


def main():
    parser = argparse.ArgumentParser(
        description="TriAttention KV cache compression for NF4 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation with TriAttention pruning
  python triattention_nf4.py --model ./Qwen3-NF4 --prompt "Explain quantum computing"

  # Aggressive compression for long generation
  python triattention_nf4.py --model ./Qwen3-NF4 --prompt "Write a story" \\
      --budget 1024 --max-tokens 8192

  # Benchmark mode (compares pruned vs full attention)
  python triattention_nf4.py --model ./Qwen3-NF4 --prompt "Explain AI" --benchmark

  # Disable pruning (full attention baseline)
  python triattention_nf4.py --model ./Qwen3-NF4 --prompt "Hello" --no-prune
""")

    # Model and generation
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help=f"Path to NF4 model directory (default: {MODEL_PATH})")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt text")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Maximum tokens to generate (default: {MAX_TOKENS})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")

    # TriAttention parameters
    parser.add_argument("--budget", type=int, default=KV_BUDGET,
                        help=f"Max tokens in KV cache (default: {KV_BUDGET})")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help=f"Pruning interval in tokens (default: {WINDOW_SIZE})")
    parser.add_argument("--recent-tokens", type=int, default=RECENT_TOKENS,
                        help=f"Recent tokens to always keep (default: {RECENT_TOKENS})")
    parser.add_argument("--sink-tokens", type=int, default=SINK_TOKENS,
                        help=f"Initial sink tokens to always keep (default: {SINK_TOKENS})")
    parser.add_argument("--no-prune", action="store_true",
                        help="Disable pruning (full attention baseline)")

    # Sampling
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature (default: {TEMPERATURE})")
    parser.add_argument("--top-p", type=float, default=TOP_P,
                        help=f"Nucleus sampling threshold (default: {TOP_P})")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Top-k sampling (default: {TOP_K})")
    parser.add_argument("--min-p", type=float, default=MIN_P,
                        help=f"Min-p sampling (default: {MIN_P})")
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY,
                        help=f"Repetition penalty (default: {REPETITION_PENALTY})")

    # Modes
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparing pruned vs baseline")
    parser.add_argument("--benchmark-tokens", type=int, default=512,
                        help="Tokens to generate in benchmark mode (default: 512)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming output")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose/debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load model
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device

    # Detect architecture
    arch_info = get_model_arch_info(model)

    # Prepare prompt
    input_ids = prepare_prompt(tokenizer, args.prompt, args.system_prompt).to(device)
    logger.info(f"Prompt: {input_ids.shape[1]} tokens")

    # Extract RoPE frequencies
    rope_inv_freq = get_rope_frequencies(model, arch_info)

    # Calibrate
    calibrator = TriAttentionCalibrator(model, arch_info)
    calibration_data = calibrator.calibrate(input_ids)

    # Create scorer
    config = model.config
    scorer = TriAttentionScorer(
        calibration_data=calibration_data,
        rope_inv_freq=rope_inv_freq,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        rotary_dim=arch_info["rotary_dim"],
    )

    if args.benchmark:
        # ── Benchmark mode ──────────────────────────────────────────────
        bench_tokens = args.benchmark_tokens
        logger.info(f"Benchmark mode: generating {bench_tokens} tokens each run")

        # Run 1: With pruning
        logger.info("--- Run 1: TriAttention pruning ---")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        pruner = TriAttentionPruner(
            scorer=scorer,
            budget=args.budget,
            window_size=args.window_size,
            recent_tokens=args.recent_tokens,
            sink_tokens=args.sink_tokens,
            full_attn_layer_indices=arch_info["full_attn_layer_indices"],
        )
        pruned_text, pruned_stats = generate_with_triattention(
            model, tokenizer, input_ids, pruner,
            max_new_tokens=bench_tokens,
            temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed, streaming=False, quiet=True,
        )

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Run 2: Without pruning (baseline)
        logger.info("--- Run 2: Full attention baseline ---")
        baseline_pruner = TriAttentionPruner(
            scorer=scorer,
            budget=999999,  # Effectively no pruning
            window_size=args.window_size,
            recent_tokens=args.recent_tokens,
            sink_tokens=args.sink_tokens,
            full_attn_layer_indices=arch_info["full_attn_layer_indices"],
        )
        baseline_text, baseline_stats = generate_with_triattention(
            model, tokenizer, input_ids, baseline_pruner,
            max_new_tokens=bench_tokens,
            temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed, streaming=False, quiet=True,
        )

        # Print benchmark report
        print("\n")
        print("=" * 70)
        print("  BENCHMARK RESULTS — TriAttention NF4")
        print("=" * 70)
        print(f"  Model:          {args.model}")
        print(f"  Prompt tokens:  {input_ids.shape[1]}")
        print(f"  Gen tokens:     {bench_tokens}")
        print(f"  KV budget:      {args.budget}")
        print(f"  Window size:    {args.window_size}")
        print("-" * 70)
        print(f"  {'Metric':<30} {'Baseline':>18} {'TriAttention':>18}")
        print("-" * 70)
        print(f"  {'Tokens generated':<30} "
              f"{baseline_stats['generated_tokens']:>18} "
              f"{pruned_stats['generated_tokens']:>18}")
        print(f"  {'Time (sec)':<30} "
              f"{baseline_stats['total_time']:>18.2f} "
              f"{pruned_stats['total_time']:>18.2f}")
        print(f"  {'Tokens/sec':<30} "
              f"{baseline_stats['tokens_per_sec']:>18.1f} "
              f"{pruned_stats['tokens_per_sec']:>18.1f}")
        print(f"  {'Final cache size':<30} "
              f"{baseline_stats['final_cache_size']:>18} "
              f"{pruned_stats['final_cache_size']:>18}")
        print(f"  {'Prune events':<30} "
              f"{baseline_stats['prune_events']:>18} "
              f"{pruned_stats['prune_events']:>18}")
        print(f"  {'Total tokens pruned':<30} "
              f"{baseline_stats['total_pruned']:>18} "
              f"{pruned_stats['total_pruned']:>18}")

        if pruned_stats['final_cache_size'] > 0 and baseline_stats['final_cache_size'] > 0:
            mem_ratio = baseline_stats['final_cache_size'] / pruned_stats['final_cache_size']
            print(f"  {'KV memory reduction':<30} {'1.0x':>18} {f'{mem_ratio:.1f}x':>18}")

        # Speed comparison
        if pruned_stats['tokens_per_sec'] > 0 and baseline_stats['tokens_per_sec'] > 0:
            speedup = pruned_stats['tokens_per_sec'] / baseline_stats['tokens_per_sec']
            print(f"  {'Throughput ratio':<30} {'1.0x':>18} {f'{speedup:.2f}x':>18}")

        # VRAM usage
        if pruned_stats['peak_vram_mb'] > 0 or baseline_stats['peak_vram_mb'] > 0:
            print(f"  {'Peak VRAM (MB)':<30} "
                  f"{baseline_stats['peak_vram_mb']:>18.0f} "
                  f"{pruned_stats['peak_vram_mb']:>18.0f}")

        print("=" * 70)

        print("\n--- Baseline output (first 500 chars) ---")
        print(baseline_text[:500])
        print("\n--- TriAttention output (first 500 chars) ---")
        print(pruned_text[:500])

    else:
        # ── Normal generation mode ──────────────────────────────────────
        if args.no_prune:
            effective_budget = 999999
            logger.info("Pruning disabled (full attention mode)")
        else:
            effective_budget = args.budget
            logger.info(f"TriAttention pruning: budget={args.budget}, "
                        f"window={args.window_size}, recent={args.recent_tokens}, "
                        f"sinks={args.sink_tokens}")

        pruner = TriAttentionPruner(
            scorer=scorer,
            budget=effective_budget,
            window_size=args.window_size,
            recent_tokens=args.recent_tokens,
            sink_tokens=args.sink_tokens,
            full_attn_layer_indices=arch_info["full_attn_layer_indices"],
        )

        try:
            text, stats = generate_with_triattention(
                model, tokenizer, input_ids, pruner,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature, top_p=args.top_p,
                top_k=args.top_k, min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed, streaming=not args.no_stream, quiet=False,
            )
            if args.no_stream:
                print(text)
            else:
                print()  # newline after streaming output

            logger.info(f"Generated {stats['generated_tokens']} tokens in "
                        f"{stats['total_time']:.2f}s "
                        f"({stats['tokens_per_sec']:.1f} tok/s)")
            if stats['prune_events'] > 0:
                logger.info(f"Pruning: {stats['prune_events']} events, "
                            f"{stats['total_pruned']} tokens evicted, "
                            f"final cache: {stats['final_cache_size']} tokens")

        except KeyboardInterrupt:
            print("\n\nGeneration interrupted.")
            sys.exit(0)


if __name__ == "__main__":
    main()
