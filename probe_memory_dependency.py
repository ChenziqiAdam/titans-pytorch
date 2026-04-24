"""
Phase 2: Per-token memory dependency measurement via ablation.

Metrics (full memory vs ablated memory):
  kl_all:      KL divergence (all memory layers disabled)
  flip_all:    prediction flip (argmax changes)
  rank_all:    rank change of target token
  kl_per_layer:   (num_tokens, num_memory_layers) per-layer KL
  flip_per_layer: (num_tokens, num_memory_layers) per-layer flip

Usage:
    python probe_memory_dependency.py \
        --checkpoint ./checkpoints/final \
        --output     ./results/memory_dependency.pt \
        [--num_batches 200] [--per_layer]
"""

import argparse
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

# ── helpers ───────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_dir, device):
    import os, glob
    from train_fineweb import build_model, TOKENIZER_NAME

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(tokenizer.vocab_size).to(device)

    # Clone model params to handle expanded (shared-memory) params from einops repeat
    for name, param in model.named_parameters():
        param.data = param.data.clone()

    weight_path = os.path.join(ckpt_dir, 'pytorch_model.bin')
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device)
        if all(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    else:
        from safetensors.torch import load_file
        sf_files = glob.glob(os.path.join(ckpt_dir, '*.safetensors'))
        if sf_files:
            state = {}
            for f in sf_files:
                state.update(load_file(f, device=str(device)))
            model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(f'No weights found in {ckpt_dir}')

    return model, tokenizer


def get_eval_batches(tokenizer, seq_len=512, batch_size=4, num_batches=200, device='cuda'):
    import glob
    total_needed = num_batches * batch_size

    sources = [
        ('HF streaming', _load_from_hf_streaming),
        ('cached parquet', _load_from_cached_parquet),
        ('wikitext', _load_wikitext),
    ]

    best = None
    for name, loader_fn in sources:
        try:
            print(f'  Trying data source: {name}...', flush=True)
            text_iter = loader_fn()
            batches = _tokenize_to_batches(text_iter, tokenizer, seq_len, batch_size, total_needed, device)
            if len(batches) >= num_batches:
                print(f'  Using {name} ({len(batches)} batches)', flush=True)
                return batches[:num_batches]
            if batches and (best is None or len(batches) > len(best)):
                best = batches
                print(f'  {name}: got {len(batches)} batches (fewer than {num_batches}, keeping as fallback)', flush=True)
            else:
                print(f'  {name}: only got {len(batches)} batches, need {num_batches}', flush=True)
        except Exception as e:
            print(f'  {name} failed: {e}', flush=True)

    if best:
        print(f'  Warning: using best available {len(best)} batches (requested {num_batches})', flush=True)
        return best
    raise RuntimeError('Could not load eval data from any source')


def _load_from_hf_streaming():
    from datasets import load_dataset
    ds = load_dataset(
        'HuggingFaceFW/fineweb-edu', name='sample-10BT',
        split='train', streaming=True,
    )
    for example in ds:
        yield example['text']


def _load_from_cached_parquet():
    import os, glob
    hf_cache = os.path.expanduser('~/.cache/huggingface/datasets')
    patterns = [
        os.path.join(hf_cache, '**/*fineweb*/**/*.parquet'),
        os.path.join(hf_cache, 'downloads/**/*.parquet'),
    ]
    parquet_files = []
    for p in patterns:
        parquet_files.extend(glob.glob(p, recursive=True))
    if not parquet_files:
        raise FileNotFoundError('No cached parquet files found')
    print(f'    Found {len(parquet_files)} cached parquet files', flush=True)
    import pyarrow.parquet as pq
    for f in sorted(parquet_files)[:5]:
        table = pq.read_table(f, columns=['text'])
        for row in table.to_pydict()['text']:
            yield row


def _load_wikitext():
    from datasets import load_dataset
    for config in ('wikitext-103-raw-v1', 'wikitext-103-v1'):
        try:
            ds = load_dataset('wikitext', config, split='test')
            for example in ds:
                if example['text'].strip():
                    yield example['text']
            return
        except Exception:
            continue
    raise RuntimeError('No wikitext-103 config found (tried raw-v1 and v1)')


def _tokenize_to_batches(text_iter, tokenizer, seq_len, batch_size, total_needed, device):
    buf = []
    chunks = []
    for text in text_iter:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buf.extend(ids)
        while len(buf) >= seq_len + 1 and len(chunks) < total_needed:
            chunk = torch.tensor(buf[:seq_len + 1], dtype=torch.long, device=device)
            buf = buf[seq_len:]
            chunks.append(chunk)
        if len(chunks) >= total_needed:
            break
    result = []
    for i in range(0, len(chunks), batch_size):
        b = chunks[i:i + batch_size]
        if len(b) == batch_size:
            result.append(torch.stack(b))
    return result


# ── metrics ───────────────────────────────────────────────────────────────────

def kl_div(p_logits, q_logits):
    """KL(p || q) per token, shapes (B, T, V) -> (B, T)."""
    p = p_logits.softmax(dim=-1)
    q = q_logits.softmax(dim=-1).clamp(min=1e-10)
    return (p * (p.log() - q.log())).sum(dim=-1)


def rank_of_target(logits, targets):
    """Rank (0-indexed) of the target token in the sorted logits. Shape: (B, T)."""
    # argsort descending
    sorted_idx = logits.argsort(dim=-1, descending=True)   # (B, T, V)
    # find position of each target
    B, T, V = logits.shape
    target_exp = targets.unsqueeze(-1)                      # (B, T, 1)
    ranks = (sorted_idx == target_exp).nonzero(as_tuple=False)
    rank_tensor = torch.zeros(B, T, dtype=torch.long, device=logits.device)
    for r in ranks:
        rank_tensor[r[0], r[1]] = r[2]
    return rank_tensor


def compute_dependency(model, batches, neural_mem_layers, per_layer, device):
    """
    Returns dict of tensors all of shape (total_tokens,) plus optional per-layer.
    """
    all_kl, all_flip, all_rank = [], [], []
    all_kl_per, all_flip_per  = [], []
    all_ids = []

    model.eval()
    with torch.no_grad():
        for batch in batches:
            x       = batch[:, :-1]
            targets = batch[:, 1:]

            # full forward
            logits_full = model(x)                        # (B, T, V)

            # full ablation
            logits_no_mem = model(x, disable_memory_layers=tuple(neural_mem_layers))

            B, T, V = logits_full.shape

            kl   = kl_div(logits_full, logits_no_mem)    # (B, T)
            flip = (logits_full.argmax(-1) != logits_no_mem.argmax(-1)).float()
            rank_full   = rank_of_target(logits_full,   targets)
            rank_no_mem = rank_of_target(logits_no_mem, targets)
            rank_change = (rank_no_mem - rank_full).float()

            all_kl.append(kl.reshape(-1).cpu())
            all_flip.append(flip.reshape(-1).cpu())
            all_rank.append(rank_change.reshape(-1).cpu())
            all_ids.append(targets.reshape(-1).cpu())

            if per_layer:
                kl_per   = []
                flip_per = []
                # Disable memory layers cumulatively (from last to first) to avoid
                # breaking the weight_residual chain between memory layers.
                # Layer order: (3, 5, 7) → ablate (7,), then (5, 7), then (3, 5, 7)
                # Each step's marginal effect = that layer's contribution.
                sorted_layers = sorted(neural_mem_layers, reverse=True)
                disabled_so_far = []
                prev_logits = logits_full
                for layer_num in sorted_layers:
                    disabled_so_far.append(layer_num)
                    lg = model(x, disable_memory_layers=tuple(disabled_so_far))
                    # Marginal KL: how much changed by adding this layer's ablation
                    kl_per.append(kl_div(prev_logits, lg).reshape(-1).cpu())
                    flip_per.append(
                        (prev_logits.argmax(-1) != lg.argmax(-1)).float().reshape(-1).cpu()
                    )
                    prev_logits = lg
                # Reverse so index 0 = layer 3, 1 = layer 5, 2 = layer 7
                kl_per.reverse()
                flip_per.reverse()
                all_kl_per.append(torch.stack(kl_per, dim=1))     # (B*T, num_mem_layers)
                all_flip_per.append(torch.stack(flip_per, dim=1))

    result = dict(
        kl_all    = torch.cat(all_kl),
        flip_all  = torch.cat(all_flip),
        rank_all  = torch.cat(all_rank),
        token_ids = torch.cat(all_ids),
    )
    if per_layer:
        result['kl_per_layer']   = torch.cat(all_kl_per,   dim=0)
        result['flip_per_layer'] = torch.cat(all_flip_per, dim=0)
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--output',      default='./results/memory_dependency.pt')
    parser.add_argument('--num_batches', type=int, default=200)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--seq_len',     type=int, default=512)
    parser.add_argument('--per_layer',   action='store_true',
                        help='also ablate each memory layer individually')
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f'Loading model from {args.checkpoint}...')
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()

    from train_fineweb import NEURAL_MEM_LAYERS
    print(f'Neural memory layers: {NEURAL_MEM_LAYERS}')

    print(f'Streaming {args.num_batches} eval batches...')
    batches = get_eval_batches(
        tokenizer, args.seq_len, args.batch_size, args.num_batches, device
    )
    print(f'Got {len(batches)} batches.')

    print('Running memory ablation...')
    results = compute_dependency(model, batches, NEURAL_MEM_LAYERS, args.per_layer, device)

    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(results, args.output)
    print(f'Saved to {args.output}')
    print(f'  kl_all:   mean={results["kl_all"].mean():.4f}')
    print(f'  flip_all: mean={results["flip_all"].mean():.4f}  (fraction of tokens changed)')
    print(f'  rank_all: mean={results["rank_all"].mean():.2f}')
    if args.per_layer:
        print(f'  kl_per_layer shape: {results["kl_per_layer"].shape}')


if __name__ == '__main__':
    main()
