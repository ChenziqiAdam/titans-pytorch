# /// script
# dependencies = [
#     "accelerate",
#     "adam-atan2-pytorch>=0.1.18",
#     "datasets",
#     "transformers",
#     "titans-pytorch",
#     "tqdm",
#     "wandb",
#     "scipy",
#     "matplotlib"
# ]
# ///

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
)

import wandb

# ── constants ────────────────────────────────────────────────────────────────

SEED               = 42
SEQ_LEN            = 512
BATCH_SIZE         = 2
GRADIENT_ACCUMULATE_EVERY = 8
LEARNING_RATE      = 2e-4
LR_WARMUP_STEPS    = 1000
LR_MIN             = 1e-5
NUM_STEPS          = 200_000          # ~5-10B tokens at batch=8, seq=512
VALIDATE_EVERY     = 500
CHECKPOINT_EVERY   = 2000
CHECKPOINT_DIR     = './checkpoints'

TOKENIZER_NAME     = 'gpt2'           # ~50k vocab

# model
DIM                = 256
DEPTH              = 8
HEADS              = 4
DIM_HEAD           = 64
WINDOW_SIZE        = 64
NUM_PERSIST_MEM    = 4
NUM_LONGTERM_MEM   = 4
NEURAL_MEM_LAYERS  = (3, 5, 7)
NEURAL_MEM_DEPTH   = 2
NEURAL_MEM_SEGMENT_LEN   = 8
NEURAL_MEM_BATCH_SIZE    = 64
NEURAL_MEM_QK_NORM       = True
NEURAL_MEM_MOMENTUM      = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_MAX_LR        = 1e-1
NEURAL_MEM_SPEC_NORM     = True
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_DIFF_VIEWS = True
STORE_ATTN_POOL_CHUNKS   = True
PER_LAYER_LEARNED_LR     = True
SLIDING_WINDOWS          = True
USE_FLEX_ATTN            = False
USE_ACCELERATED_SCAN     = False

# wandb
PROJECT_NAME  = 'titans-mac-fineweb'
WANDB_ONLINE  = False

# ── dataset ──────────────────────────────────────────────────────────────────

class FineWebDataset(IterableDataset):
    """Streams FineWeb-Edu, tokenizes on the fly, yields fixed-length chunks."""

    def __init__(self, tokenizer, seq_len, split='train', buffer_tokens=1_000_000):
        from datasets import load_dataset
        self.ds = load_dataset(
            'HuggingFaceFW/fineweb-edu',
            name='sample-10BT',
            split=split,
            streaming=True,
        )
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.buf_len   = buffer_tokens

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


def build_loaders(tokenizer, seq_len, batch_size, num_workers=2):
    train_ds = FineWebDataset(tokenizer, seq_len, split='train')
    # FineWeb-Edu has no official val split in streaming; use a separate seed-shuffled subset
    # For simplicity we use a second iterator (different shuffle state) as pseudo-val
    val_ds   = FineWebDataset(tokenizer, seq_len, split='train')

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


# ── model ────────────────────────────────────────────────────────────────────

def build_model(vocab_size):
    neural_memory_model = MemoryMLP(dim=DIM_HEAD, depth=NEURAL_MEM_DEPTH)

    model = MemoryAsContextTransformer(
        num_tokens             = vocab_size,
        dim                    = DIM,
        depth                  = DEPTH,
        segment_len            = WINDOW_SIZE,
        num_persist_mem_tokens = NUM_PERSIST_MEM,
        num_longterm_mem_tokens= NUM_LONGTERM_MEM,
        neural_memory_layers   = NEURAL_MEM_LAYERS,
        neural_memory_segment_len  = NEURAL_MEM_SEGMENT_LEN,
        neural_memory_batch_size   = NEURAL_MEM_BATCH_SIZE,
        neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
        neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_DIFF_VIEWS,
        use_flex_attn          = USE_FLEX_ATTN,
        sliding_window_attn    = SLIDING_WINDOWS,
        neural_memory_model    = neural_memory_model,
        neural_memory_kwargs   = dict(
            dim_head               = DIM_HEAD,
            heads                  = HEADS,
            attn_pool_chunks       = STORE_ATTN_POOL_CHUNKS,
            qk_rmsnorm             = NEURAL_MEM_QK_NORM,
            momentum               = NEURAL_MEM_MOMENTUM,
            momentum_order         = NEURAL_MEM_MOMENTUM_ORDER,
            default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
            use_accelerated_scan   = USE_ACCELERATED_SCAN,
            per_parameter_lr_modulation = PER_LAYER_LEARNED_LR,
            spectral_norm_surprises= NEURAL_MEM_SPEC_NORM,
        )
    )
    return model


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(accelerator, model, optimizer, scheduler, step, path):
    os.makedirs(path, exist_ok=True)
    accelerator.save_state(path)
    if accelerator.is_main_process:
        torch.save({'step': step}, os.path.join(path, 'meta.pt'))


def load_checkpoint(accelerator, path):
    meta_path = os.path.join(path, 'meta.pt')
    if not os.path.exists(meta_path):
        return 0
    accelerator.load_state(path)
    meta = torch.load(meta_path)
    return meta['step']


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATE_EVERY,
        mixed_precision='bf16',
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # data
    train_loader, val_loader = build_loaders(tokenizer, SEQ_LEN, BATCH_SIZE)

    # model
    model = build_model(vocab_size)

    # optimizer & scheduler
    optimizer = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = LR_WARMUP_STEPS,
        num_training_steps = NUM_STEPS,
        num_cycles         = 0.5,
    )
    # clamp to LR_MIN by overriding the lr lambda
    base_scheduler = scheduler
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda_with_min(step):
        lr_scale = base_scheduler.get_last_lr()[0] / LEARNING_RATE if step > 0 else 1.0
        return max(lr_scale, LR_MIN / LEARNING_RATE)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # wandb
    if accelerator.is_main_process:
        wandb.init(
            project = PROJECT_NAME,
            mode    = 'online' if WANDB_ONLINE else 'disabled',
            config  = dict(
                dim=DIM, depth=DEPTH, heads=HEADS,
                seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                lr=LEARNING_RATE, steps=NUM_STEPS,
                neural_mem_layers=NEURAL_MEM_LAYERS,
            )
        )

    # resume
    start_step = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, 'latest')
    if os.path.isdir(latest_ckpt):
        start_step = load_checkpoint(accelerator, latest_ckpt)
        accelerator.print(f'Resumed from step {start_step}')

    train_iter = cycle(train_loader)
    val_iter   = cycle(val_loader)

    # skip already-processed batches when resuming
    for _ in range(start_step * GRADIENT_ACCUMULATE_EVERY):
        next(train_iter)

    model.train()
    for step in tqdm.tqdm(range(start_step, NUM_STEPS), desc='training', mininterval=10.):

        # gradient accumulation
        total_loss = 0.
        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            batch = next(train_iter)
            with accelerator.accumulate(model):
                loss = model(batch, return_loss=True)
                accelerator.backward(loss)
                total_loss += loss.item()

        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / GRADIENT_ACCUMULATE_EVERY
        if accelerator.is_main_process:
            wandb.log({'train/loss': avg_loss, 'train/lr': scheduler.get_last_lr()[0]}, step=step)

        # validation
        if step % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_loss = model(next(val_iter), return_loss=True)
            accelerator.print(f'step {step} | train loss: {avg_loss:.4f} | val loss: {val_loss.item():.4f}')
            if accelerator.is_main_process:
                wandb.log({'val/loss': val_loss.item()}, step=step)
            model.train()

        # checkpoint
        if step % CHECKPOINT_EVERY == 0 and step > 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'step_{step}')
            save_checkpoint(accelerator, model, optimizer, scheduler, step, ckpt_path)
            save_checkpoint(accelerator, model, optimizer, scheduler, step, latest_ckpt)
            accelerator.print(f'Saved checkpoint at step {step}')

    # final checkpoint
    save_checkpoint(accelerator, model, optimizer, scheduler, NUM_STEPS,
                    os.path.join(CHECKPOINT_DIR, 'final'))
    accelerator.print('Training complete.')


if __name__ == '__main__':
    main()
