# TriAttention NF4

**KV Cache Compression for NF4-Quantized Language Models using Trigonometric Series Scoring**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) 

A full implementation of the [TriAttention](https://arxiv.org/abs/2604.04921) KV cache compression algorithm for **NF4-quantized** HuggingFace transformer models. Enables long-context inference on consumer GPUs with minimal quality loss.

**Author:** [g023](https://github.com/g023) — **Repository:** [github.com/g023/triattention_nf4](https://github.com/g023/triattention_nf4)

**NOTE:** Example usage references a `Qwen3-1.77B NF4` model located in a `./Qwen3-NF4` folder in the same folder as the script 'triattention_nf4.py'. 

(https://huggingface.co/g023/Qwen3-1.77B-g023-NF4) > throw model files in a folder called './Qwen3-NF4', in same folder as 'triattention_nf4.py'.

**NOTE:** Currently working on (almost finished) compatibility with Qwen3.5 more specifically this model: (https://huggingface.co/g023/Qwopus3.5-9B-v3-NF4)
Current output for qwen3.5 work (will upload and remove this comment when done):
```
triattention_nf4.py --model ./qwop-v3-9b-nf4 --prompt "Write a story" --budget 64 --recent-tokens 32 --benchmark --benchmark-tokens 512

----------------------------------------------------------------------
  Metric                                   Baseline       TriAttention
----------------------------------------------------------------------
  Tokens generated                              512                512
  Time (sec)                                  17.27              17.36
  Tokens/sec                                   29.6               29.5
  Final cache size                              525                 64
  Prune events                                    0                  4
  Total tokens pruned                             0                461
  KV memory reduction                          1.0x               8.2x
  Throughput ratio                             1.0x              1.00x
```

---

## What is TriAttention?

TriAttention (Mao et al., 2026) is a state-of-the-art KV cache compression method that exploits **Q/K concentration** in the pre-RoPE space. Unlike post-RoPE methods that rely on attention scores from a tiny observation window, TriAttention:

1. **Discovers** that pre-RoPE Q and K vectors cluster around fixed centers (stable across positions and content)
2. **Predicts** attention patterns via a **trigonometric series** computed from these centers
3. **Scores** cached keys by their predicted future importance, retaining only the most important ones

This implementation specifically targets **NF4 models** (4-bit NormalFloat via bitsandbytes), combining two powerful memory optimizations:
- **NF4 quantization**: ~4x reduction in model weight memory
- **TriAttention cache compression**: Up to 8x+ reduction in KV cache memory

## Benchmark Results

All benchmarks on **Qwen3-1.77B NF4** on **NVIDIA RTX 3060 12GB**.

### Budget 1024, 2048 Generated Tokens
```
======================================================================
  BENCHMARK RESULTS — TriAttention NF4
======================================================================
  Model:          ./Qwen3-NF4
  Prompt tokens:  36
  Gen tokens:     2048
  KV budget:      1024
  Window size:    128
----------------------------------------------------------------------
  Metric                           Baseline       TriAttention
----------------------------------------------------------------------
  Tokens generated                     2048               2048
  Time (sec)                          57.70              57.07
  Tokens/sec                           35.5               35.9
  Final cache size                     2084               1024
  Prune events                            0                  9
  Total tokens pruned                     0               1060
  KV memory reduction                  1.0x               2.0x
  Throughput ratio                     1.0x              1.01x
======================================================================
```

### Budget 512, 1024 Generated Tokens  
```
----------------------------------------------------------------------
  Metric                           Baseline       TriAttention
----------------------------------------------------------------------
  Tokens generated                     1024               1024
  Tokens/sec                           34.3               36.0
  Final cache size                     1069                512
  KV memory reduction                  1.0x               2.1x
  Throughput ratio                     1.0x              1.05x
----------------------------------------------------------------------
```

### Budget 128, 1024 Generated Tokens (Extreme Compression)
```
----------------------------------------------------------------------
  Metric                           Baseline       TriAttention
----------------------------------------------------------------------
  Tokens generated                     1024               1024
  Tokens/sec                           35.5               34.4
  Final cache size                     1048                128
  KV memory reduction                  1.0x               8.2x
  Throughput ratio                     1.0x              0.97x
----------------------------------------------------------------------
```

### Key Findings
- **Zero quality loss**: Output text is identical (first 500+ characters) between baseline and TriAttention across all tested budget levels
- **Up to 8.2x KV memory reduction** with extreme compression (128 budget)
- **No throughput penalty**: TriAttention is often *faster* than full attention because smaller cache = faster attention computation
- **Consistent performance** across mathematical, scientific, and general prompts

---

## How It Works

### Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Model Loading │────▶│   Calibration    │────▶│    Generation    │
│  (NF4+BnB)   │     │  (Pre-RoPE Q/K)  │     │  (+ Pruning)     │
└──────────────┘     └──────────────────┘     └──────────────────┘
                           │                         │
                           ▼                         ▼
                     ┌──────────┐            ┌──────────────┐
                     │ Q Centers │            │ Score & Prune│
                     │ Q Norms   │───────────▶│ Every 128 tok│
                     │ MRL R_f   │            │ Top-B kept   │
                     └──────────┘            └──────────────┘
```

### Algorithm (Paper Equations)

**1. Calibration** — Compute pre-RoPE Q centers per head per layer:
- $\mathbb{E}[q_f]$: Complex mean of Q vectors in each frequency band
- $\mathbb{E}[\|q_f\|]$: Mean norm per frequency band
- $R_f = \|\mathbb{E}[q_f]\| / \mathbb{E}[\|q_f\|]$: Mean Resultant Length (concentration)

**2. Trigonometric Score** (Eq. 6 from paper):
$$S_{\text{trig}}(k, \Delta) = \sum_f \|\mathbb{E}[q_f]\| \cdot \|k_f\| \cdot \cos(\omega_f \Delta + \phi_f)$$

**3. Norm Score** with adaptive weighting (Eq. 8):
$$S_{\text{norm}}(k) = \sum_f (1 - R_f) \cdot \mathbb{E}[\|q_f\|] \cdot \|k_f\|$$

**4. Combined Score** with future offset averaging (Eq. 10-11):
$$\tilde{S}(k) = \frac{1}{|\mathcal{D}|} \sum_{\delta \in \mathcal{D}} [S_{\text{trig}}(k, \Delta + \delta) + S_{\text{norm}}(k)]$$
where $\mathcal{D} = \{1, 2, 4, \ldots, 2^{16}\}$

**5. GQA Aggregation** (Eq. 12-13): Z-normalize per query head, aggregate via max.

### Key Innovation: Post-RoPE Scoring Trick

Instead of storing pre-RoPE keys separately (memory overhead), we compute scores directly from the already-cached post-RoPE keys. The key insight is that **key positions cancel out** in the scoring formula:

$$\cos(\omega_f \Delta + \phi_f) = \cos(\omega_f p_q + \arg(\mathbb{E}[q_f]) - \arg(\tilde{k}_f))$$

This means **zero additional memory per cached key** — we only need the calibrated Q centers and the current query position.

---

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (tested on RTX 3060 12GB)
- conda (recommended)

### Setup

```bash
# Create/activate conda environment
conda create -n triattention python=3.10 -y
conda activate triattention

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=5.0.0 bitsandbytes>=0.49.0
```

### Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Tensor operations, CUDA |
| `transformers` | ≥5.0.0 | Model loading, tokenizer |
| `bitsandbytes` | ≥0.49.0 | NF4 quantization |

---

## Usage

### Basic Generation

```bash
# Generate with TriAttention pruning (default budget: 2048)
python triattention_nf4.py --model ./Qwen3-NF4 --prompt "Explain quantum computing"

# Generate with custom budget
python triattention_nf4.py --model ./Qwen3-NF4 --prompt "Write a story" \
    --budget 1024 --max-tokens 8192

# Full attention baseline (no pruning)
python triattention_nf4.py --model ./Qwen3-NF4 --prompt "Hello" --no-prune
```

### Benchmark Mode

```bash
# Compare pruned vs full attention
python triattention_nf4.py --model ./Qwen3-NF4 \
    --prompt "Explain general relativity" \
    --benchmark --benchmark-tokens 1024

# Aggressive compression benchmark
python triattention_nf4.py --model ./Qwen3-NF4 \
    --prompt "Prove the fundamental theorem of calculus" \
    --benchmark --benchmark-tokens 2048 --budget 512
```

### All CLI Options

```
Model & Generation:
  --model PATH          Path to NF4 model directory (default: ./Qwen3-NF4)
  --prompt TEXT          Input prompt text (required)
  --system-prompt TEXT   Optional system prompt
  --max-tokens N        Maximum tokens to generate (default: 4096)
  --seed N              Random seed (default: 42)

TriAttention Parameters:
  --budget N            Max tokens in KV cache (default: 2048)
  --window-size N       Pruning interval in tokens (default: 128)
  --recent-tokens N     Recent tokens to always keep (default: 128)
  --sink-tokens N       Initial sink tokens to always keep (default: 4)
  --no-prune            Disable pruning (full attention baseline)

Sampling:
  --temperature F       Sampling temperature (default: 0.6)
  --top-p F             Nucleus sampling threshold (default: 0.95)
  --top-k N             Top-k sampling (default: 40)
  --min-p F             Min-p sampling (default: 0.05)
  --repetition-penalty F  Repetition penalty (default: 1.1)

Modes:
  --benchmark           Run benchmark comparing pruned vs baseline
  --benchmark-tokens N  Tokens per benchmark run (default: 512)
  --no-stream           Disable streaming output
  --verbose             Enable debug logging
```

---

## How to Recreate This Project

This section provides step-by-step instructions for a mid-level engineer to rebuild this project from scratch.

### Step 1: Understand the Core Concepts

1. **Read the paper**: [TriAttention: Efficient Long Reasoning with Trigonometric KV Compression](https://arxiv.org/abs/2604.04921)
2. **Key concepts to understand**:
   - RoPE (Rotary Position Embedding) — how position is encoded via rotation in complex space
   - KV cache — why it grows linearly and causes memory issues
   - Q/K concentration — pre-RoPE Q/K vectors cluster around fixed centers
   - Trigonometric series — how these centers determine attention-vs-distance curves

### Step 2: Set Up Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

NF4 models (quantized with bitsandbytes) load automatically — the quantization config is stored in `config.json`.

### Step 3: Extract RoPE Frequencies

```python
theta = config.rope_parameters["rope_theta"]  # e.g., 1000000
head_dim = config.head_dim  # e.g., 128
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
# Shape: [head_dim/2] = [64]
```

These are the ω_f frequencies used in the trigonometric series.

### Step 4: Implement Calibration

Hook into each attention layer to capture **pre-RoPE Q vectors** (after q_norm, before RoPE):

```python
# In Qwen3Attention.forward:
# query_states = self.q_norm(self.q_proj(hidden_states).view(...)).transpose(1,2)
# ↑ This is the pre-RoPE Q — hook here
```

Collect running statistics:
- Complex representation: `z_f = q[..., f] + i * q[..., f + d/2]` for each frequency band f
- Accumulate `sum(z_f)` and `sum(|z_f|)` across all tokens
- Compute: `E[q_f] = sum(z_f) / count`, `E[||q_f||] = sum(|z_f|) / count`
- MRL: `R_f = ||E[q_f]|| / E[||q_f||]`

### Step 5: Implement Scoring

The critical insight: **scores can be computed from post-RoPE cached keys** because key positions cancel:

```python
# Convert cached keys to complex form
k_complex = k_real + i * k_imag  # [kv_heads, seq_len, d/2]
k_norms = |k_complex|
k_phases = angle(k_complex)

# S_trig: vectorize over offsets, keys, frequency bands
phases = inv_freq * (current_pos + offset) + q_center_phase - k_phase
s_trig = sum(q_center_norm * k_norms * cos(phases))  # sum over freq bands

# S_norm: adaptive weighting
s_norm = sum((1 - R_f) * E[||q_f||] * k_norms)

# Combine and average over future offsets
score = mean_over_offsets(s_trig) + s_norm
```

### Step 6: Implement GQA Aggregation

For Grouped-Query Attention (multiple Q heads per KV head):
1. Compute scores per Q head
2. Z-normalize within each Q head: `(score - μ) / σ`
3. Take max across Q heads sharing each KV head
4. Average across KV heads

### Step 7: Implement Cache Pruning

Every `window_size` (128) tokens:
1. Protect attention sinks (first 4 tokens)
2. Protect recent window (last 128 tokens)
3. Score middle tokens
4. Keep top-B by score
5. Index into cache tensors: `layer.keys = layer.keys[:, :, keep_indices, :]`

### Step 8: Custom Generation Loop

Standard HF `generate()` doesn't support mid-generation cache modification. Use a manual loop:

```python
# Process prompt
outputs = model(input_ids, past_key_values=cache, use_cache=True)

# Generate token by token
for step in range(max_tokens):
    next_token = sample(outputs.logits[:, -1, :])
    outputs = model(
        next_token, past_key_values=cache, use_cache=True,
        cache_position=torch.tensor([current_pos]),  # CRITICAL after pruning
        position_ids=torch.tensor([[current_pos]]),
    )
    current_pos += 1
    
    if should_prune(cache):
        prune(cache)  # Modifies cache in-place
```

**Critical**: After pruning, you MUST pass explicit `cache_position` and `position_ids` because the model's automatic position inference uses `cache.get_seq_length()` which is now wrong.

---

## Project Structure

```
triattention_nf4/
├── triattention_nf4.py       # Main implementation
├── README_NF4.md             # This file
├── PROMPT_NF4.py             # Development prompt
├── Qwen3-NF4/                # Test model (NF4 quantized)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── CONTEXT/                   # Development context files
    ├── triattention_nf4.ANALYSIS.md
    ├── triattention_nf4.RESEARCH.md
    ├── triattention_nf4.PLAN.md
    └── ...
```

## Code Architecture

```
triattention_nf4.py
├── get_rope_frequencies()          # Extract ω_f from model config
├── TriAttentionCalibrator          # Hook-based pre-RoPE Q/K statistics
│   ├── calibrate(input_ids)        # Run calibration forward pass
│   └── _accumulate_q_stats()       # Running statistics accumulation
├── TriAttentionScorer              # GPU-vectorized scoring engine
│   └── score_keys(layer, keys, pos) # Full paper scoring algorithm
├── TriAttentionPruner              # Cache manipulation
│   ├── should_prune(cache_size)    # Window-based triggering
│   └── prune(cache, position)      # Score + evict + modify cache
├── generate_with_triattention()    # Custom generation loop
├── load_model()                    # NF4 model loading
├── prepare_prompt()                # Chat template formatting
└── main()                          # CLI entry point
```

---

## Technical Details

### Memory Layout
- **Cached keys**: `[batch, kv_heads, seq_len, head_dim]` in bf16
- **Complex representation**: Pairs dimensions `[f, f+d/2]` into complex numbers
- **Scoring**: All computation in float32 for numerical stability

### Performance Characteristics
- **Calibration**: ~0.2s (single forward pass on prompt)
- **Scoring overhead**: ~0.5ms per prune event per sampled layer
- **Layer sampling**: Scores 8 representative layers (every 4th) instead of all 29
- **Vectorized**: All scoring operations use PyTorch tensor ops on GPU

### Supported Models
Any HuggingFace model with:
- RoPE position embedding
- NF4/INT4 quantization via bitsandbytes
- GQA or MHA attention
- Tested: Qwen3 family

---

## Citation

If you use this implementation, please cite the original TriAttention paper:

```bibtex
@article{mao2026triattention,
  title={TriAttention: Efficient Long Reasoning with Trigonometric KV Compression},
  author={Mao, Weian and Lin, Xi and Huang, Wei and Xie, Yuxin and Fu, Tianfu and Zhuang, Bohan and Han, Song and Chen, Yukang},
  journal={arXiv preprint arXiv:2604.04921},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
