"""Phase 1 (base competency): plain next-token pretraining of the External stack alone
(no Internal model, no interleaving) on a small ClimbMix shard slice, per the approved
plan. Produces the checkpoint Phase 2 (phase2_train.py) loads and continues training with
illation switched on.

Deliberately uses the same lightweight ExternalStack class (external_model.py) that
InterleavedStack wraps in Phase 2 -- not nanochat's full GPT class with its value-
embeddings/smear/backout/window-pattern/vocab-padding extras -- so a Phase 1 checkpoint
loads directly into Phase 2 with no architecture mismatch. Uses a plain AdamW warmup+cosine
schedule rather than porting nanochat's Muon-specific schedule, since we aren't using
Muon; this project's own history (see CLAUDE project memory) already showed plain AdamW
works fine at this scale without needing that schedule.

Tokenization: reuses our own already-trained ByteLevelBPE tokenizer (tokenizer.py,
vocab=8000, trained on our own corpus) for ClimbMix text too, rather than pulling in
nanochat's rustbpe/tiktoken tokenizer chain (avoided earlier in this project for
dependency reasons -- see project memory). ClimbMix shards are raw text (parquet 'text'
column), so any tokenizer works; reusing ours keeps Phase 1 and Phase 2 vocab identical.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from tokenizer import load as load_tokenizer
from external_model import ExternalConfig, ExternalStack
from nanochat.dataset import parquets_iter_batched, list_parquet_files, DATA_DIR

HERE = Path(__file__).parent


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_token_stream(tok, split, max_tokens=None):
    """Yields a flat stream of token ids from ClimbMix text batches."""
    n_yielded = 0
    for texts in parquets_iter_batched(split):
        for text in texts:
            ids = tok.encode(text).ids
            for i in ids:
                yield i
                n_yielded += 1
                if max_tokens is not None and n_yielded >= max_tokens:
                    return


def build_token_tensor(tok, split, max_tokens):
    buf = torch.empty(max_tokens, dtype=torch.long)
    n = 0
    for i in iter_token_stream(tok, split, max_tokens):
        buf[n] = i
        n += 1
        if n >= max_tokens:
            break
    return buf[:n]


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def lm_loss(model, x, y):
    x_emb = model.embed(x)
    x_final, _ = model.run_stack(x_emb)
    logits = model.logits(x_final)
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def lr_multiplier(step, total_steps, warmup_steps, warmdown_frac=0.65, final_frac=0.05):
    warmdown_start = int(total_steps * (1 - warmdown_frac))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < warmdown_start:
        return 1.0
    progress = (step - warmdown_start) / max(total_steps - warmdown_start, 1)
    return 1.0 - (1.0 - final_frac) * progress


@torch.no_grad()
def estimate_val_loss(model, val_data, block_size, device, iters=20, batch_size=8):
    model.eval()
    losses = [lm_loss(model, *get_batch(val_data, block_size, batch_size, device)).item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def run(n_shards, steps, block_size, n_layer, n_head, n_embd, out_dir, log_every=25, eval_every=200, max_val_tokens=2_000_000):
    device = get_device()
    print(f"device={device}")

    print(f"downloading {n_shards} ClimbMix shard(s) (+1 val shard) if not already present...")
    from nanochat.dataset import download_single_file, DATA_DIR, MAX_SHARD
    import os
    os.makedirs(DATA_DIR, exist_ok=True)  # download_single_file assumes this exists (normally
    # created by dataset.py's own CLI __main__ block, which we bypass calling it as a library)
    for i in range(n_shards):
        download_single_file(i)
    download_single_file(MAX_SHARD)

    tok = load_tokenizer()
    vocab_size = tok.get_vocab_size()

    shard_paths = list_parquet_files()
    train_shard_bytes = sum(Path(p).stat().st_size for p in shard_paths[:-1])
    approx_tokens = int(train_shard_bytes / 4.5)  # ~4.5 bytes/token rough estimate for English BPE
    print(f"train shards: {len(shard_paths)-1}, ~{train_shard_bytes/1e6:.1f}MB raw, ~{approx_tokens/1e6:.1f}M tokens estimated")

    max_train_tokens = min(approx_tokens, 60_000_000)  # cap for this run's time budget
    print(f"tokenizing up to {max_train_tokens/1e6:.1f}M train tokens...")
    t0 = time.time()
    train_data = build_token_tensor(tok, "train", max_train_tokens)
    print(f"  got {len(train_data)/1e6:.2f}M train tokens in {time.time()-t0:.1f}s")
    val_data = build_token_tensor(tok, "val", max_val_tokens)
    print(f"  got {len(val_data)/1e6:.2f}M val tokens")

    config = ExternalConfig(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_kv_head=n_head, n_embd=n_embd, max_seq_len=block_size)
    model = ExternalStack(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M | tokens:params ratio ~= {len(train_data)/max(n_params,1):.1f}")

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    base_lr = opt.param_groups[0]["lr"]
    warmup_steps = min(100, steps // 20 or 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    t0 = time.time()

    for step in range(1, steps + 1):
        opt.param_groups[0]["lr"] = base_lr * lr_multiplier(step - 1, steps, warmup_steps)
        x, y = get_batch(train_data, block_size, batch_size=16, device=device)
        loss = lm_loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:6d}/{steps} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e} | elapsed {elapsed:7.1f}s")
            history.append({"step": step, "train_loss": loss.item(), "elapsed": elapsed})

        if step % eval_every == 0 or step == steps:
            val_loss = estimate_val_loss(model, val_data, block_size, device)
            ppl = math.exp(min(val_loss, 20))
            print(f"  >> val_loss {val_loss:.4f} (ppl {ppl:.2f})")
            if history and history[-1]["step"] == step:
                history[-1]["val_loss"] = val_loss
            else:
                history.append({"step": step, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "config": config.__dict__, "step": step, "val_loss": val_loss},
                           out_dir / "best.pt")

    torch.save({"model": model.state_dict(), "config": config.__dict__, "step": steps}, out_dir / "final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done. best val_loss={best_val:.4f}. saved to {out_dir}")
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=12)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--out", type=str, default=str(HERE / "runs" / "v4_phase1"))
    args = ap.parse_args()
    run(args.n_shards, args.steps, args.block_size, args.n_layer, args.n_head, args.n_embd, args.out)
