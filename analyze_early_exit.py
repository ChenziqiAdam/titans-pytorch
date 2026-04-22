"""
Phase 4 analysis: Early-exit layer with vs without memory.

Correlates exit-layer reduction with Phase 1 difficulty and Phase 2 dependency.

Usage:
    python analyze_early_exit.py \
        --early_exit  ./results/early_exit.pt \
        --difficulty  ./results/difficulty.pt \
        --dependency  ./results/memory_dependency.pt \
        --output_dir  ./results/analysis_exit
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_and_align(*paths):
    """Load .pt files, align on min token count."""
    dicts = [torch.load(p) for p in paths]
    n = min(len(d['token_ids']) for d in dicts)
    return dicts, n


def partial_spearman(x, y, z):
    r_xy = stats.spearmanr(x, y).statistic
    r_xz = stats.spearmanr(x, z).statistic
    r_yz = stats.spearmanr(y, z).statistic
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-9:
        return 0.
    return (r_xy - r_xz * r_yz) / denom


def decile_plot(x, y, xlabel, ylabel, title, path):
    deciles = np.percentile(x, np.arange(0, 101, 10))
    bin_idx = np.digitize(x, deciles[1:-1])
    means, centers = [], []
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        means.append(y[mask].mean())
        centers.append((deciles[b] + deciles[b + 1]) / 2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, means, 'o-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved {path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--early_exit', required=True)
    parser.add_argument('--difficulty', required=True)
    parser.add_argument('--dependency', required=True)
    parser.add_argument('--output_dir', default='./results/analysis_exit')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading data...', flush=True)
    (ee, diff, dep), n = load_and_align(args.early_exit, args.difficulty, args.dependency)

    exit_mem = ee['exit_layer_mem'][:n].numpy().astype(float)
    exit_nomem = ee['exit_layer_nomem'][:n].numpy().astype(float)
    exit_red = ee['exit_reduction'][:n].numpy().astype(float)

    metric_a = diff['metric_a'][:n].numpy()
    metric_b = diff['metric_b'][:n].numpy()
    metric_c = diff['metric_c'][:n].numpy()

    kl_all = dep['kl_all'][:n].numpy()
    flip_all = dep['flip_all'][:n].numpy()

    token_ids = diff['token_ids'][:n].numpy().astype(int)
    counts = np.bincount(token_ids, minlength=50257).astype(float) + 1
    log_freq = np.log(counts / counts.sum())[token_ids]

    # ── 1. Summary stats ─────────────────────────────────────────────────
    print(f'\n=== Early Exit Summary (n={n:,} tokens) ===', flush=True)
    print(f'  Mean exit layer WITH memory:    {exit_mem.mean():.2f}', flush=True)
    print(f'  Mean exit layer WITHOUT memory:  {exit_nomem.mean():.2f}', flush=True)
    print(f'  Mean exit reduction:             {exit_red.mean():.2f}', flush=True)
    print(f'  Tokens where memory helps (red>0): {(exit_red > 0).mean():.1%}', flush=True)
    print(f'  Tokens where memory hurts (red<0): {(exit_red < 0).mean():.1%}', flush=True)

    # ── 2. Correlations: exit_reduction vs difficulty ─────────────────────
    diff_metrics = [
        ('Cosine instability (A)', metric_a),
        ('Entropy (B)', metric_b),
        ('Settling layer (C)', metric_c),
    ]
    dep_metrics = [
        ('KL divergence', kl_all),
        ('Prediction flip', flip_all),
    ]

    print(f'\n=== Spearman r: Exit reduction vs Difficulty ===', flush=True)
    for name, arr in diff_metrics:
        r, p = stats.spearmanr(exit_red, arr)
        pr = partial_spearman(exit_red, arr, log_freq)
        print(f'  {name}: r={r:.4f} (p={p:.3e}), partial_r(freq)={pr:.4f}', flush=True)

    # ── 3. Correlations: exit_reduction vs dependency ─────────────────────
    print(f'\n=== Spearman r: Exit reduction vs Dependency ===', flush=True)
    for name, arr in dep_metrics:
        r, p = stats.spearmanr(exit_red, arr)
        pr = partial_spearman(exit_red, arr, log_freq)
        print(f'  {name}: r={r:.4f} (p={p:.3e}), partial_r(freq)={pr:.4f}', flush=True)

    # ── 4. Correlations: exit layers (mem/nomem) vs difficulty ────────────
    print(f'\n=== Spearman r: Exit layer (mem) vs Difficulty ===', flush=True)
    for name, arr in diff_metrics:
        r, _ = stats.spearmanr(exit_mem, arr)
        print(f'  {name}: r={r:.4f}', flush=True)

    print(f'\n=== Spearman r: Exit layer (nomem) vs Difficulty ===', flush=True)
    for name, arr in diff_metrics:
        r, _ = stats.spearmanr(exit_nomem, arr)
        print(f'  {name}: r={r:.4f}', flush=True)

    # ── 5. Decile plots ──────────────────────────────────────────────────
    for name, arr in diff_metrics:
        safe = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        decile_plot(
            arr, exit_red, name, 'Exit reduction (layers)',
            f'Exit reduction vs {name}',
            os.path.join(args.output_dir, f'exit_red_vs_{safe}.png'),
        )
        decile_plot(
            arr, exit_mem, name, 'Exit layer (with memory)',
            f'Exit layer (mem) vs {name}',
            os.path.join(args.output_dir, f'exit_mem_vs_{safe}.png'),
        )
        decile_plot(
            arr, exit_nomem, name, 'Exit layer (no memory)',
            f'Exit layer (no mem) vs {name}',
            os.path.join(args.output_dir, f'exit_nomem_vs_{safe}.png'),
        )

    # ── 6. Histogram of exit reduction ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    vals, edges = np.histogram(exit_red, bins=np.arange(exit_red.min() - 0.5, exit_red.max() + 1.5, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, vals / vals.sum(), width=0.8)
    ax.set_xlabel('Exit reduction (layers)')
    ax.set_ylabel('Fraction of tokens')
    ax.set_title('Distribution of exit-layer reduction (memory vs no memory)')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'exit_reduction_histogram.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved {path}', flush=True)

    print(f'\nAll outputs saved to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
