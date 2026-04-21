"""
Phase 1: Per-token difficulty measurement.

Metrics:
  A - cosine instability:  1 - min_l cos_sim(h_l, h_{l+1})   (high = hard)
  B - final entropy:       entropy of softmax(last-layer logits)
  C - settling layer:      earliest layer where argmax matches final-layer argmax

Usage:
    python probe_difficulty.py \
        --checkpoint ./checkpoints/final \
        --output     ./results/difficulty.pt \
        [--num_batches 200]
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# ── helpers ───────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_dir, device):
    """Load model + tokenizer saved by train_fineweb.py via accelerator.save_state."""
    import os, glob
    from accelerate import Accelerator

    # build the same architecture as train_fineweb.py
    from train_fineweb import build_model, TOKENIZER_NAME, NEURAL_MEM_LAYERS

    tokenizer  = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(tokenizer.vocab_size).to(device)

    # Clone model params to handle expanded (shared-memory) params from einops repeat
    for name, param in model.named_parameters():
        param.data = param.data.clone()

    # load weights
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
    """Stream a small FineWeb-Edu slice for evaluation."""
    from datasets import load_dataset
    ds = load_dataset(
        'HuggingFaceFW/fineweb-edu',
        name='sample-10BT',
        split='train',
        streaming=True,
    )

    buf = []
    batches = []
    for example in ds:
        ids = tokenizer.encode(example['text'], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buf.extend(ids)
        while len(buf) >= seq_len + 1 and len(batches) * batch_size < num_batches * batch_size:
            chunk = torch.tensor(buf[:seq_len + 1], dtype=torch.long, device=device)
            buf   = buf[seq_len:]
            batches.append(chunk)
            if len(batches) >= num_batches * batch_size:
                break
        if len(batches) >= num_batches * batch_size:
            break

    # group into batches
    result = []
    for i in range(0, len(batches), batch_size):
        b = batches[i:i+batch_size]
        if len(b) == batch_size:
            result.append(torch.stack(b))
    return result


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_difficulty(model, batches, device):
    """
    Returns dict of tensors, each (total_tokens,):
      metric_a: cosine instability (max change between adjacent layers)
      metric_b: final-layer prediction entropy
      metric_c: settling layer (first layer whose argmax == final argmax)
      token_ids: the actual target token ids
    """
    all_a, all_b, all_c, all_ids = [], [], [], []

    model.eval()
    with torch.no_grad():
        for batch in batches:
            # batch: (B, seq_len+1)
            x       = batch[:, :-1]   # (B, seq_len)
            targets = batch[:, 1:]    # (B, seq_len)

            logits, hidden_states = model(x, return_hidden_states=True)
            # logits: (B, seq_len, vocab)
            # hidden_states: list of L tensors, each (B, seq_len, dim)

            B, T, V = logits.shape
            L = len(hidden_states)

            # ── Metric A: cosine instability ──────────────────────────────
            # For each adjacent pair of layers, compute cos_sim per token
            # metric_a[t] = max over l of (1 - cos_sim(h_l[t], h_{l+1}[t]))
            stacked = torch.stack(hidden_states, dim=0)  # (L, B, T, D)
            cos_sim = F.cosine_similarity(
                stacked[:-1], stacked[1:], dim=-1
            )  # (L-1, B, T)
            instability = 1. - cos_sim          # high = more change
            metric_a = instability.max(dim=0).values  # (B, T)

            # ── Metric B: final-layer entropy ─────────────────────────────
            probs    = logits.softmax(dim=-1)   # (B, T, V)
            log_p    = probs.log().clamp(min=-1e9)
            metric_b = -(probs * log_p).sum(dim=-1)  # (B, T)

            # ── Metric C: settling layer ──────────────────────────────────
            final_pred = logits.argmax(dim=-1)  # (B, T)

            # per-layer predictions using the shared norm + lm_head
            metric_c = torch.full((B, T), L, dtype=torch.long, device=device)

            # check layers from last to first to find earliest stable layer
            agreed = torch.ones(B, T, dtype=torch.bool, device=device)
            for l_idx in range(L - 1, -1, -1):
                h = hidden_states[l_idx]          # (B, T, D)
                layer_logits = model.to_logits(model.norm(h))
                layer_pred   = layer_logits.argmax(dim=-1)  # (B, T)
                matches_final = (layer_pred == final_pred)
                # a token settles at l_idx if it matches and all later layers also agree
                agreed = agreed & matches_final
                metric_c[agreed] = l_idx

            # flatten batch & time dims
            all_a.append(metric_a.reshape(-1).cpu())
            all_b.append(metric_b.reshape(-1).cpu())
            all_c.append(metric_c.reshape(-1).cpu())
            all_ids.append(targets.reshape(-1).cpu())

    return dict(
        metric_a   = torch.cat(all_a),
        metric_b   = torch.cat(all_b),
        metric_c   = torch.cat(all_c).float(),
        token_ids  = torch.cat(all_ids),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--output',      default='./results/difficulty.pt')
    parser.add_argument('--num_batches', type=int, default=200)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--seq_len',     type=int, default=512)
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f'Loading model from {args.checkpoint}...')
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    print(f'Model loaded. Vocab size: {tokenizer.vocab_size}')

    print(f'Streaming {args.num_batches} batches of eval data...')
    batches = get_eval_batches(
        tokenizer, args.seq_len, args.batch_size, args.num_batches, device
    )
    print(f'Got {len(batches)} batches.')

    print('Computing token difficulty metrics...')
    results = compute_difficulty(model, batches, device)

    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(results, args.output)
    print(f'Saved to {args.output}')
    print(f'  metric_a shape: {results["metric_a"].shape}')
    print(f'  metric_b shape: {results["metric_b"].shape}')
    print(f'  metric_c shape: {results["metric_c"].shape}')
    print(f'  mean settling layer: {results["metric_c"].mean():.2f}')
    print(f'  mean entropy: {results["metric_b"].mean():.4f}')


if __name__ == '__main__':
    main()
