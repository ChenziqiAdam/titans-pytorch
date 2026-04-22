# Experiment: Do tokens needing more layers also depend more on memory?

This document describes the experiment pipeline for testing whether token difficulty (how many layers a token needs to settle) correlates with memory dependency (how much removing the neural memory module changes predictions).

## Model Configuration

| Parameter | Value |
|-----------|-------|
| dim | 256 |
| depth | 8 layers |
| heads | 4 |
| dim_head | 64 |
| neural_memory_layers | (3, 5, 7) |
| memory model | MemoryMLP(dim=64, depth=2) |
| tokenizer | GPT-2 (~50k vocab) |
| seq_len | 512 |
| training data | FineWeb-Edu sample-10BT |

## Pipeline Overview

```
Phase 0: Train model          -> ./checkpoints/final/
Phase 0.5: Sanity check       -> confirms memory is useful
Phase 1: Probe difficulty      -> ./results/difficulty.pt
Phase 2: Probe dependency      -> ./results/memory_dependency.pt
Phase 3: Analyze correlation   -> ./results/analysis/
Phase 4: Analyze early exit    -> ./results/early_exit.pt, ./results/analysis_exit/
```

---

## Phase 0: Training (`train_fineweb.py`)

Trains a MAC Transformer with neural memory on FineWeb-Edu.

```bash
# Single GPU
uv run python train_fineweb.py

# Multi-GPU via accelerate
uv run accelerate launch train_fineweb.py
```

**Key settings** (edit constants at top of file):
- `NUM_STEPS = 10_000` (~80M tokens, proof-of-concept)
- `CHECKPOINT_EVERY = 1000` saves to `./checkpoints/step_N/`
- `VALIDATE_EVERY = 200` logs validation loss
- `WANDB_ONLINE = False` (set `True` to log to W&B)

**Output**: Checkpoints in `./checkpoints/`, final checkpoint in `./checkpoints/final/`.

---

## Phase 0.5: Sanity Check (`phase0_sanity_check.py`)

Compares perplexity with vs without memory. If the gap is < 0.5 ppl points, the memory module isn't contributing meaningfully and you should train longer or tune hyperparameters.

```bash
uv run python phase0_sanity_check.py --checkpoint ./checkpoints/final
```

**Output**: Prints perplexity with memory, without memory, and the difference.

---

## Phase 1: Token Difficulty (`probe_difficulty.py`)

Measures how "hard" each token is for the model using three metrics:

| Metric | Name | Meaning |
|--------|------|---------|
| A | Cosine instability | `1 - min_l cos_sim(h_l, h_{l+1})` — high = representation changes a lot between layers |
| B | Final entropy | Entropy of softmax(last-layer logits) — high = model is uncertain |
| C | Settling layer | Earliest layer where argmax matches final prediction and stays correct — high = needs more layers |

```bash
uv run python probe_difficulty.py \
    --checkpoint ./checkpoints/final \
    --output ./results/difficulty.pt \
    --num_batches 200 \
    --batch_size 4 \
    --seq_len 512 \
    --device cuda
```

**Output**: `./results/difficulty.pt` containing:
- `metric_a`: (N,) float — cosine instability per token
- `metric_b`: (N,) float — entropy per token
- `metric_c`: (N,) float — settling layer per token
- `token_ids`: (N,) int — target token IDs

**How to read results**: Higher values = harder tokens. Metric C directly measures "how many layers does this token need."

---

## Phase 2: Memory Dependency (`probe_memory_dependency.py`)

Measures how much each token's prediction depends on the neural memory by comparing full model output vs output with memory disabled (ablation).

| Metric | Name | Meaning |
|--------|------|---------|
| kl_all | KL divergence | KL(full || no-mem) — high = memory changes distribution a lot |
| flip_all | Prediction flip | Whether argmax changes when memory is removed — binary |
| rank_all | Rank change | How much the target token's rank worsens without memory — high = memory helps this token |

```bash
uv run python probe_memory_dependency.py \
    --checkpoint ./checkpoints/final \
    --output ./results/memory_dependency.pt \
    --num_batches 200 \
    --batch_size 4 \
    --seq_len 512 \
    --device cuda

# Optional: also measure per-layer memory contribution
uv run python probe_memory_dependency.py \
    --checkpoint ./checkpoints/final \
    --output ./results/memory_dependency.pt \
    --per_layer \
    --device cuda
```

**Output**: `./results/memory_dependency.pt` containing:
- `kl_all`: (N,) float — KL divergence per token
- `flip_all`: (N,) float — prediction flip (0 or 1) per token
- `rank_all`: (N,) float — rank change per token
- `token_ids`: (N,) int — target token IDs
- `kl_per_layer`: (N, 3) float — per-memory-layer KL (only with `--per_layer`)
- `flip_per_layer`: (N, 3) float — per-memory-layer flip (only with `--per_layer`)

**How to read results**: Higher KL / more flips / larger rank change = token depends more on memory.

---

## Phase 3: Correlation Analysis (`analyze_correlation.py`)

Correlates Phase 1 difficulty metrics with Phase 2 dependency metrics. This is the core test of the hypothesis.

```bash
uv run python analyze_correlation.py \
    --difficulty ./results/difficulty.pt \
    --dependency ./results/memory_dependency.pt \
    --output_dir ./results/analysis \
    --n_boot 1000
```

**Output** (in `./results/analysis/`):

| File | Description |
|------|-------------|
| `correlation_heatmap.png` | 3x3 Spearman correlation matrix (difficulty x dependency) |
| `diff_metric_*_vs_dep_*.png` | Decile scatter plots: tokens binned by difficulty, showing mean dependency |
| `per_layer_kl_*.png` | Per-memory-layer correlation with difficulty (only if `--per_layer` was used in Phase 2) |

**Printed output includes**:
- Spearman correlation matrix with p-values
- Bootstrap 95% confidence intervals
- Partial correlations controlling for token frequency
- Partial correlations controlling for sequence position

**How to interpret**:
- Positive Spearman r between difficulty and dependency = harder tokens depend more on memory (hypothesis supported)
- Partial correlations check if the relationship holds after removing confounds (frequency, position)
- If r > 0 but partial_r ≈ 0, the correlation was driven by the confound, not a true relationship

---

## Phase 4: Early Exit Analysis (`probe_early_exit.py` + `analyze_early_exit.py`)

Tests a stronger version of the hypothesis: does memory let the model settle on the correct prediction at an earlier layer?

### Step 1: Probe exit layers

```bash
uv run python probe_early_exit.py \
    --checkpoint ./checkpoints/final \
    --output ./results/early_exit.pt \
    --num_batches 200 \
    --batch_size 4 \
    --seq_len 512 \
    --device cuda
```

**Output**: `./results/early_exit.pt` containing:
- `exit_layer_mem`: (N,) int — exit layer with memory enabled
- `exit_layer_nomem`: (N,) int — exit layer with memory disabled
- `exit_reduction`: (N,) int — `nomem - mem` (positive = memory helps exit earlier)
- `token_ids`: (N,) int — target token IDs

### Step 2: Analyze

```bash
uv run python analyze_early_exit.py \
    --early_exit ./results/early_exit.pt \
    --difficulty ./results/difficulty.pt \
    --dependency ./results/memory_dependency.pt \
    --output_dir ./results/analysis_exit
```

**Output** (in `./results/analysis_exit/`):

| File | Description |
|------|-------------|
| `exit_reduction_histogram.png` | Distribution of exit-layer reduction (how many layers earlier with memory) |
| `exit_layer_joint_heatmap.png` | 2D heatmap: exit layer with memory (x) vs without memory (y) per token |
| `exit_layer_comparison_hist.png` | Side-by-side histograms of exit layer distributions |
| `exit_red_vs_*.png` | Exit reduction binned by difficulty decile |
| `exit_mem_vs_*.png` | Exit layer (with mem) binned by difficulty |
| `exit_nomem_vs_*.png` | Exit layer (no mem) binned by difficulty |

**How to interpret**:
- If most mass in the heatmap is below the diagonal, memory lets tokens exit earlier
- If most mass is on the diagonal, memory doesn't affect convergence speed
- Exit reduction histogram centered at 0 = memory doesn't accelerate settling

---

## Full Pipeline (copy-paste)

```bash
# 1. Train
uv run python train_fineweb.py

# 2. Sanity check
uv run python phase0_sanity_check.py --checkpoint ./checkpoints/final

# 3. Run probes (can run in parallel on separate GPUs)
uv run python probe_difficulty.py --checkpoint ./checkpoints/final --output ./results/difficulty.pt
uv run python probe_memory_dependency.py --checkpoint ./checkpoints/final --output ./results/memory_dependency.pt --per_layer
uv run python probe_early_exit.py --checkpoint ./checkpoints/final --output ./results/early_exit.pt

# 4. Analyze
uv run python analyze_correlation.py --difficulty ./results/difficulty.pt --dependency ./results/memory_dependency.pt --output_dir ./results/analysis
uv run python analyze_early_exit.py --early_exit ./results/early_exit.pt --difficulty ./results/difficulty.pt --dependency ./results/memory_dependency.pt --output_dir ./results/analysis_exit
```

---

## Model Modifications (`titans_pytorch/mac_transformer.py`)

Two parameters were added to `MemoryAsContextTransformer.forward()`:

1. **`return_hidden_states=False`** — When `True`, returns `(logits, hidden_states)` where `hidden_states` is a list of L tensors, each `(B, T, D)`, one per transformer layer. Used by Phase 1 and Phase 4.

2. **`disable_memory_layers=None`** — Accepts a tuple of 1-indexed layer numbers (matching `neural_memory_layers` config). When set, the neural memory module is skipped at those layers. Used by Phase 2 and Phase 4.

Both are fully backward-compatible: default values preserve original behavior.

---

## Data Loading

All probe scripts use a 3-source fallback for evaluation data:
1. **HuggingFace streaming** — `fineweb-edu` sample-10BT (requires internet)
2. **Cached parquet files** — from `~/.cache/huggingface/datasets/` (works offline if previously downloaded)
3. **WikiText-103** — `wikitext-103-raw-v1` test split (smallest, always available via HF)

---

## Key Findings (from our runs)

**Phase 3 (Correlation)**: Weak but statistically significant positive correlations (Spearman r = 0.03–0.16) between token difficulty and memory dependency. Partial correlations controlling for frequency remain positive. The hypothesis is directionally supported: harder tokens do depend slightly more on memory.

**Phase 4 (Early Exit)**: Memory has negligible effect on exit layers (mean reduction ≈ 0.03 layers). The joint heatmap shows most tokens on the diagonal. Conclusion: memory changes *what* the model predicts for hard tokens, but not *when* it settles on that prediction. Memory is a content signal, not a convergence accelerator.
