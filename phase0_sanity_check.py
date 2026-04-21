# /// script
# dependencies = [
#     "datasets",
#     "transformers",
#     "titans-pytorch",
#     "tqdm",
# ]
# ///

"""
Phase 0 sanity check: compare perplexity WITH memory vs WITHOUT memory.
If gap < 0.5 ppl points, memory is not being meaningfully used.

Usage:
    uv run python phase0_sanity_check.py --checkpoint ./checkpoints/final
    uv run python phase0_sanity_check.py --checkpoint ./checkpoints/step_10000
"""

import argparse
import math
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm

from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# ── must match train_fineweb.py constants ────────────────────────────────────
SEQ_LEN              = 512
BATCH_SIZE           = 4
TOKENIZER_NAME       = 'gpt2'
DIM                  = 256
DEPTH                = 8
HEADS                = 4
DIM_HEAD             = 64
WINDOW_SIZE          = 64
NUM_PERSIST_MEM      = 4
NUM_LONGTERM_MEM     = 4
NEURAL_MEM_LAYERS    = (3, 5, 7)
NEURAL_MEM_DEPTH     = 2
NEURAL_MEM_SEGMENT_LEN    = 32
NEURAL_MEM_BATCH_SIZE     = 32
NEURAL_MEM_QK_NORM        = True
NEURAL_MEM_MOMENTUM       = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_MAX_LR         = 1e-1
NEURAL_MEM_SPEC_NORM      = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_DIFF_VIEWS = True
STORE_ATTN_POOL_CHUNKS    = True
PER_LAYER_LEARNED_LR      = True
SLIDING_WINDOWS           = True
USE_FLEX_ATTN             = False
USE_ACCELERATED_SCAN      = False

NUM_EVAL_BATCHES     = 50   # ~50 * 4 * 512 = ~100k tokens, fast enough


class FineWebDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len, buffer_tokens=500_000):
        from datasets import load_dataset
        self.ds = load_dataset(
            'HuggingFaceFW/fineweb-edu',
            name='sample-10BT',
            split='train',
            streaming=True,
        )
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __iter__(self):
        buf = []
        for example in self.ds:
            ids = self.tokenizer.encode(example['text'], add_special_tokens=False)
            ids.append(self.tokenizer.eos_token_id)
            buf.extend(ids)
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf   = buf[self.seq_len:]
                yield torch.tensor(chunk, dtype=torch.long)


def build_model(vocab_size):
    from copy import deepcopy
    neural_memory_model = MemoryMLP(dim=DIM_HEAD, depth=NEURAL_MEM_DEPTH)
    model = MemoryAsContextTransformer(
        num_tokens                  = vocab_size,
        dim                         = DIM,
        depth                       = DEPTH,
        segment_len                 = WINDOW_SIZE,
        num_persist_mem_tokens      = NUM_PERSIST_MEM,
        num_longterm_mem_tokens     = NUM_LONGTERM_MEM,
        neural_memory_layers        = NEURAL_MEM_LAYERS,
        neural_memory_segment_len   = NEURAL_MEM_SEGMENT_LEN,
        neural_memory_batch_size    = NEURAL_MEM_BATCH_SIZE,
        neural_mem_weight_residual  = NEURAL_MEM_WEIGHT_RESIDUAL,
        neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_DIFF_VIEWS,
        use_flex_attn               = USE_FLEX_ATTN,
        sliding_window_attn         = SLIDING_WINDOWS,
        neural_memory_model         = neural_memory_model,
        neural_memory_kwargs        = dict(
            dim_head                     = DIM_HEAD,
            heads                        = HEADS,
            attn_pool_chunks             = STORE_ATTN_POOL_CHUNKS,
            qk_rmsnorm                   = NEURAL_MEM_QK_NORM,
            momentum                     = NEURAL_MEM_MOMENTUM,
            momentum_order               = NEURAL_MEM_MOMENTUM_ORDER,
            default_step_transform_max_lr= NEURAL_MEM_MAX_LR,
            use_accelerated_scan         = USE_ACCELERATED_SCAN,
            per_parameter_lr_modulation  = PER_LAYER_LEARNED_LR,
            spectral_norm_surprises      = NEURAL_MEM_SPEC_NORM,
        ),
    )
    return model


@torch.no_grad()
def compute_ppl(model, loader, device, disable_memory_layers=None, max_batches=NUM_EVAL_BATCHES):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(tqdm(loader, total=max_batches, desc='eval')):
        if i >= max_batches:
            break
        batch = batch.to(device)
        loss = model(batch, return_loss=True, disable_memory_layers=disable_memory_layers)
        n_tokens = batch.shape[0] * (batch.shape[1] - 1)
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens
    ppl = math.exp(total_loss / total_tokens)
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint dir, e.g. ./checkpoints/final or ./checkpoints/step_10000')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Build model and load weights
    model = build_model(vocab_size)

    # Load checkpoint — accelerate saves model weights as model.safetensors or pytorch_model.bin
    import os
    ckpt_candidates = [
        os.path.join(args.checkpoint, 'model.safetensors'),
        os.path.join(args.checkpoint, 'pytorch_model.bin'),
        os.path.join(args.checkpoint, 'model.pt'),
    ]
    loaded = False
    for ckpt_path in ckpt_candidates:
        if os.path.exists(ckpt_path):
            print(f'Loading weights from {ckpt_path}')
            if ckpt_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location='cpu')
            # Accelerate wraps the model; strip 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            loaded = True
            break

    if not loaded:
        raise FileNotFoundError(f'No model weights found in {args.checkpoint}. Tried: {ckpt_candidates}')

    model = model.to(device)

    # Eval data — use a fresh stream (different offset from training)
    eval_ds = FineWebDataset(tokenizer, SEQ_LEN)
    # Skip first 500k tokens to avoid overlap with early training data
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, num_workers=2)

    print(f'\nEvaluating WITH memory ({NUM_EVAL_BATCHES} batches)...')
    ppl_with_mem = compute_ppl(model, eval_loader, device, disable_memory_layers=None)

    print(f'\nEvaluating WITHOUT memory ({NUM_EVAL_BATCHES} batches)...')
    ppl_no_mem = compute_ppl(
        model, eval_loader, device,
        disable_memory_layers=tuple(NEURAL_MEM_LAYERS)
    )

    gap = ppl_no_mem - ppl_with_mem
    print('\n' + '='*50)
    print(f'  PPL  with memory : {ppl_with_mem:.3f}')
    print(f'  PPL  no   memory : {ppl_no_mem:.3f}')
    print(f'  Gap  (no - with) : {gap:.3f}')
    print('='*50)

    if gap < 0.5:
        print('\n[WARNING] Gap < 0.5 ppl — memory is NOT contributing meaningfully.')
        print('  Consider: more training steps, higher NEURAL_MEM_MAX_LR, or more memory layers.')
    else:
        print('\n[OK] Memory is contributing meaningfully. Proceed to Phase 1.')


if __name__ == '__main__':
    main()
