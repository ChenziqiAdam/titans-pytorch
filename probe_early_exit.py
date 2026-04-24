"""
Phase 4: Early-exit layer analysis — with vs without memory.

For each token, find the earliest layer whose hidden-state prediction
matches the final layer's top-1 prediction ("exit layer").
Compare exit layers with memory enabled vs disabled.

Hypothesis: memory lets tokens settle on the correct prediction earlier,
and the reduction is larger for harder tokens.

Outputs (saved to .pt):
  exit_layer_mem:    (N,) int — exit layer with memory
  exit_layer_nomem:  (N,) int — exit layer without memory
  exit_reduction:    (N,) int — how many layers earlier with memory
  token_ids:         (N,) int — target token ids

Usage:
    python probe_early_exit.py \
        --checkpoint ./checkpoints/final \
        --output     ./results/early_exit.pt \
        [--num_batches 200]
"""

import argparse
import os
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer


# ── helpers (shared with other probes) ────────────────────────────────────────

def load_model_from_checkpoint(ckpt_dir, device):
    import glob
    from train_fineweb import build_model, TOKENIZER_NAME

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(tokenizer.vocab_size).to(device)

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

    for name, loader_fn in sources:
        try:
            print(f'  Trying data source: {name}...', flush=True)
            text_iter = loader_fn()
            batches = _tokenize_to_batches(text_iter, tokenizer, seq_len, batch_size, total_needed, device)
            if len(batches) >= num_batches:
                return batches[:num_batches]
            print(f'  {name}: only got {len(batches)} batches, need {num_batches}', flush=True)
        except Exception as e:
            print(f'  {name} failed: {e}', flush=True)

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
    import glob
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


# ── core computation ──────────────────────────────────────────────────────────

def compute_exit_layer(model, x, hidden_states):
    """
    Given hidden_states (list of L tensors, each (B, T, D)), find the earliest
    layer whose argmax prediction matches the final layer's argmax.
    Returns: (B, T) int tensor of exit layers (0-indexed).
    """
    # Final prediction from full logits
    final_h = hidden_states[-1]
    final_logits = model.to_logits(model.norm(final_h))
    final_pred = final_logits.argmax(dim=-1)  # (B, T)

    B, T = final_pred.shape
    L = len(hidden_states)

    exit_layer = torch.full((B, T), L - 1, dtype=torch.long, device=x.device)

    # Scan from last to first; exit_layer = earliest layer that agrees
    # with final pred and all subsequent layers also agree
    agreed = torch.ones(B, T, dtype=torch.bool, device=x.device)
    for l_idx in range(L - 1, -1, -1):
        h = hidden_states[l_idx]
        layer_logits = model.to_logits(model.norm(h))
        layer_pred = layer_logits.argmax(dim=-1)
        matches = (layer_pred == final_pred)
        agreed = agreed & matches
        exit_layer[agreed] = l_idx

    return exit_layer


def compute_early_exit(model, batches, neural_mem_layers, device):
    """Run forward with and without memory, compute exit layers for both."""
    all_exit_mem = []
    all_exit_nomem = []
    all_ids = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(batches):
            if batch_idx % 10 == 0:
                print(f'  batch {batch_idx}/{len(batches)}', flush=True)

            x = batch[:, :-1]
            targets = batch[:, 1:]

            # With memory
            logits_mem, hidden_mem = model(x, return_hidden_states=True)
            exit_mem = compute_exit_layer(model, x, hidden_mem)

            # Without memory
            logits_nomem, hidden_nomem = model(
                x, return_hidden_states=True,
                disable_memory_layers=tuple(neural_mem_layers),
            )
            exit_nomem = compute_exit_layer(model, x, hidden_nomem)

            all_exit_mem.append(exit_mem.reshape(-1).cpu())
            all_exit_nomem.append(exit_nomem.reshape(-1).cpu())
            all_ids.append(targets.reshape(-1).cpu())

    exit_mem = torch.cat(all_exit_mem)
    exit_nomem = torch.cat(all_exit_nomem)

    return dict(
        exit_layer_mem=exit_mem,
        exit_layer_nomem=exit_nomem,
        exit_reduction=(exit_nomem - exit_mem),
        token_ids=torch.cat(all_ids),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='./results/early_exit.pt')
    parser.add_argument('--num_batches', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f'Loading model from {args.checkpoint}...', flush=True)
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()

    from train_fineweb import NEURAL_MEM_LAYERS
    print(f'Neural memory layers: {NEURAL_MEM_LAYERS}', flush=True)

    print(f'Streaming {args.num_batches} eval batches...', flush=True)
    batches = get_eval_batches(
        tokenizer, args.seq_len, args.batch_size, args.num_batches, device
    )
    print(f'Got {len(batches)} batches.', flush=True)

    print('Computing early-exit layers (with vs without memory)...', flush=True)
    results = compute_early_exit(model, batches, NEURAL_MEM_LAYERS, device)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(results, args.output)

    em = results['exit_layer_mem'].float()
    en = results['exit_layer_nomem'].float()
    red = results['exit_reduction'].float()

    print(f'Saved to {args.output}', flush=True)
    print(f'  Mean exit layer WITH memory:    {em.mean():.2f}', flush=True)
    print(f'  Mean exit layer WITHOUT memory:  {en.mean():.2f}', flush=True)
    print(f'  Mean exit reduction (nomem - mem): {red.mean():.2f}', flush=True)
    print(f'  Tokens where memory helps (reduction > 0): {(red > 0).float().mean():.1%}', flush=True)
    print(f'  Tokens where memory hurts (reduction < 0): {(red < 0).float().mean():.1%}', flush=True)


if __name__ == '__main__':
    main()
