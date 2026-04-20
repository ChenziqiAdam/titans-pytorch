"""
Phase 3: Correlation analysis between token difficulty and memory dependency.

Usage:
    python analyze_correlation.py \
        --difficulty ./results/difficulty.pt \
        --dependency ./results/memory_dependency.pt \
        --output_dir ./results/analysis
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ── load & align ──────────────────────────────────────────────────────────────

def load_and_align(diff_path, dep_path):
    diff = torch.load(diff_path)
    dep  = torch.load(dep_path)

    # align on token_ids length (take min)
    n = min(len(diff['token_ids']), len(dep['token_ids']))
    result = {}
    for k, v in diff.items():
        result[f'diff_{k}'] = v[:n].numpy()
    for k, v in dep.items():
        result[f'dep_{k}'] = v[:n].numpy() if v.ndim == 1 else v[:n].numpy()
    return result, n


# ── token frequency ────────────────────────────────────────────────────────────

def compute_log_frequency(token_ids, vocab_size=50257):
    counts = np.bincount(token_ids.astype(int), minlength=vocab_size).astype(float)
    counts += 1   # Laplace smoothing
    freq = counts / counts.sum()
    log_freq = np.log(freq[token_ids.astype(int)])
    return log_freq


def partial_correlation(x, y, z):
    """Partial Spearman correlation of x,y controlling for z."""
    res_x = stats.spearmanr(x, z).statistic
    res_y = stats.spearmanr(y, z).statistic
    # partial r via formula
    r_xy  = stats.spearmanr(x, y).statistic
    r_xz  = res_x
    r_yz  = res_y
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-9:
        return 0.
    return (r_xy - r_xz * r_yz) / denom


# ── correlation matrix ────────────────────────────────────────────────────────

DIFFICULTY_KEYS = ['diff_metric_a', 'diff_metric_b', 'diff_metric_c']
DIFF_LABELS     = ['Cosine instability (A)', 'Entropy (B)', 'Settling layer (C)']
DEP_KEYS        = ['dep_kl_all', 'dep_flip_all', 'dep_rank_all']
DEP_LABELS      = ['KL divergence', 'Prediction flip', 'Rank change']


def correlation_matrix(data, diff_keys, dep_keys):
    """Returns (len(diff_keys), len(dep_keys)) matrix of Spearman correlations."""
    mat = np.zeros((len(diff_keys), len(dep_keys)))
    for i, dk in enumerate(diff_keys):
        for j, depk in enumerate(dep_keys):
            r, _ = stats.spearmanr(data[dk], data[depk])
            mat[i, j] = r
    return mat


def bootstrap_ci(x, y, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for Spearman correlation."""
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rs.append(r)
    rs = np.array(rs)
    lo = np.percentile(rs, (1 - ci) / 2 * 100)
    hi = np.percentile(rs, (1 + ci) / 2 * 100)
    return lo, hi


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_heatmap(mat, row_labels, col_labels, title, path):
    fig, ax = plt.subplots(figsize=(len(col_labels) * 2.5, len(row_labels) * 2))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=30, ha='right')
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved {path}')


def plot_decile_scatter(diff_arr, dep_arr, diff_label, dep_label, path):
    """Bin tokens by difficulty decile, plot mean dependency."""
    deciles = np.percentile(diff_arr, np.arange(0, 101, 10))
    bin_idx = np.digitize(diff_arr, deciles[1:-1])
    means, stds, centers = [], [], []
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        means.append(dep_arr[mask].mean())
        stds.append(dep_arr[mask].std())
        centers.append((deciles[b] + deciles[b+1]) / 2)
    means = np.array(means); stds = np.array(stds); centers = np.array(centers)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(centers, means, yerr=stds, fmt='o-', capsize=4)
    ax.set_xlabel(diff_label); ax.set_ylabel(f'Mean {dep_label}')
    ax.set_title(f'{diff_label} vs {dep_label} (binned by decile)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved {path}')


def plot_per_layer_heatmap(kl_per_layer, diff_arr, diff_label, mem_layers, path):
    """For each memory layer, compute Spearman r with difficulty."""
    rs = []
    for l_idx in range(kl_per_layer.shape[1]):
        r, _ = stats.spearmanr(diff_arr, kl_per_layer[:, l_idx])
        rs.append(r)
    fig, ax = plt.subplots(figsize=(len(rs) * 1.2 + 1, 3))
    ax.bar(range(len(rs)), rs)
    ax.set_xticks(range(len(rs)))
    ax.set_xticklabels([f'Layer {l}' for l in mem_layers])
    ax.set_ylabel('Spearman r')
    ax.set_title(f'Per-layer KL vs {diff_label}')
    ax.axhline(0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved {path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty',  required=True)
    parser.add_argument('--dependency',  required=True)
    parser.add_argument('--output_dir',  default='./results/analysis')
    parser.add_argument('--n_boot',      type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading data...')
    data, n = load_and_align(args.difficulty, args.dependency)
    print(f'Aligned on {n} tokens.')

    token_ids = data['diff_token_ids'].astype(int)
    log_freq  = compute_log_frequency(token_ids)
    seq_pos   = np.tile(np.arange(512), n // 512 + 1)[:n]

    # ── 1. correlation matrix ─────────────────────────────────────────────
    print('\n=== Spearman correlation matrix ===')
    mat = correlation_matrix(data, DIFFICULTY_KEYS, DEP_KEYS)
    print('            ', '  '.join(f'{l:>14}' for l in DEP_LABELS))
    for i, dl in enumerate(DIFF_LABELS):
        print(f'{dl:30s}', '  '.join(f'{mat[i,j]:14.4f}' for j in range(len(DEP_LABELS))))

    plot_heatmap(
        mat, DIFF_LABELS, DEP_LABELS,
        'Spearman r: Difficulty vs Memory Dependency',
        os.path.join(args.output_dir, 'correlation_heatmap.png')
    )

    # ── 2. bootstrap CIs ──────────────────────────────────────────────────
    print('\n=== Bootstrap 95% CIs (Spearman r) ===')
    for dk, dl in zip(DIFFICULTY_KEYS, DIFF_LABELS):
        for depk, depl in zip(DEP_KEYS, DEP_LABELS):
            r, p = stats.spearmanr(data[dk], data[depk])
            lo, hi = bootstrap_ci(data[dk], data[depk], n_boot=args.n_boot)
            print(f'  {dl} x {depl}: r={r:.4f} p={p:.3e} CI=[{lo:.4f}, {hi:.4f}]')

    # ── 3. partial correlations controlling for frequency & position ──────
    print('\n=== Partial Spearman r (controlling for log-frequency) ===')
    for dk, dl in zip(DIFFICULTY_KEYS, DIFF_LABELS):
        for depk, depl in zip(DEP_KEYS, DEP_LABELS):
            pr = partial_correlation(data[dk], data[depk], log_freq)
            print(f'  {dl} x {depl}: partial_r={pr:.4f}')

    print('\n=== Partial Spearman r (controlling for sequence position) ===')
    for dk, dl in zip(DIFFICULTY_KEYS, DIFF_LABELS):
        for depk, depl in zip(DEP_KEYS, DEP_LABELS):
            pr = partial_correlation(data[dk], data[depk], seq_pos)
            print(f'  {dl} x {depl}: partial_r={pr:.4f}')

    # ── 4. decile scatter plots ───────────────────────────────────────────
    for dk, dl in zip(DIFFICULTY_KEYS, DIFF_LABELS):
        for depk, depl in zip(DEP_KEYS, DEP_LABELS):
            fname = f'{dk}_vs_{depk}.png'.replace('/', '_')
            plot_decile_scatter(
                data[dk], data[depk], dl, depl,
                os.path.join(args.output_dir, fname)
            )

    # ── 5. per-layer KL heatmap (if available) ────────────────────────────
    if 'dep_kl_per_layer' in data:
        from train_fineweb import NEURAL_MEM_LAYERS
        for dk, dl in zip(DIFFICULTY_KEYS, DIFF_LABELS):
            fname = f'per_layer_kl_{dk}.png'.replace('/', '_')
            plot_per_layer_heatmap(
                data['dep_kl_per_layer'], data[dk], dl,
                list(NEURAL_MEM_LAYERS),
                os.path.join(args.output_dir, fname)
            )

    print(f'\nAll outputs saved to {args.output_dir}')


if __name__ == '__main__':
    main()
