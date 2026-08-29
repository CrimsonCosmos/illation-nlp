import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from illation import THOUGHT_END, THOUGHT_START, LOOKUP, IllationConfig, IllationEngine, IllationStats

HERE = Path(__file__).parent
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
SEQUENCE_LEN = 512  # our training context window (independent of the model's own 8192 max)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model_and_tokenizer(device):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    added = tok.add_special_tokens({"additional_special_tokens": [THOUGHT_START, THOUGHT_END, LOOKUP]})
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    if added:
        model.resize_token_embeddings(len(tok))
    model.to(device)
    return model, tok


def load_data(tok, val_fraction=0.02):
    corpus_path = HERE / "data" / "corpus.txt"
    text = corpus_path.read_text(encoding="utf-8")
    ids = tok.encode(text, add_special_tokens=False)
    ids = torch.tensor(ids, dtype=torch.long)
    n_val = int(len(ids) * val_fraction)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    print(f"tokens: train={len(train_ids):,} val={len(val_ids):,}")
    return train_ids, val_ids


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def lm_loss_fn(model, x, y):
    logits = model(x).logits
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def estimate_val_loss(model, val_data, block_size, device, iters=20, batch_size=8):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val_data, block_size, batch_size, device)
        losses.append(lm_loss_fn(model, x, y).item())
    model.train()
    return sum(losses) / len(losses)


def run(mode: str, steps: int, log_every: int, eval_every: int, out_dir: Path):
    device = get_device()
    print(f"mode={mode} device={device}")

    model, tok = build_model_and_tokenizer(device)
    model.train()
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Gentle fine-tuning schedule, not from-scratch-pretraining rates: the model already
    # knows English going in, we're only nudging it (and, in illation mode, teaching the
    # extra thought/lookup mechanism), so a small constant-ish LR with short warmup is
    # what avoids blowing away the pretrained weights.
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95), weight_decay=0.01)
    warmup_steps = min(50, steps // 10 or 1)

    def lr_at(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    train_data, val_data = load_data(tok)

    illation_cfg = IllationConfig()
    illation_engine = IllationEngine(model, tok, illation_cfg, device, max_seq_len=SEQUENCE_LEN) if mode == "illation" else None
    illation_every = 4       # run one illation batch every N optimizer steps
    illation_batch = 8       # number of rows imagined together per illation episode
    stats = IllationStats()

    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")
    best_step = 0
    t0 = time.time()
    base_lr = opt.param_groups[0]["lr"]

    for step in range(1, steps + 1):
        opt.param_groups[0]["lr"] = base_lr * lr_at(step - 1)
        x, y = get_batch(train_data, SEQUENCE_LEN, batch_size=8, device=device)
        lm_loss = lm_loss_fn(model, x, y)
        loss = lm_loss

        illation_loss_val = None
        if illation_engine is not None and step % illation_every == 0:
            nb = min(illation_batch, x.size(0))
            rows = random.sample(range(x.size(0)), nb)
            full_ids = torch.stack([torch.cat([x[b], y[b, -1:]]) for b in rows])  # (nb, L+1) contiguous stream
            headroom = (1 + illation_cfg.max_thought_len
                        + illation_cfg.max_lookups_per_thought * (illation_cfg.max_lookup_tokens + 2)
                        + 1 + illation_cfg.future_window)
            pos = random.randint(SEQUENCE_LEN // 4, SEQUENCE_LEN - headroom - 2)
            illation_loss = illation_engine.run_batch(full_ids, pos, stats)
            if illation_loss is not None:
                loss = loss + illation_loss
                illation_loss_val = float(illation_loss)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            msg = f"step {step:5d} | lm_loss {lm_loss.item():.4f} | elapsed {elapsed:6.1f}s"
            if illation_loss_val is not None:
                msg += (f" | illation_loss {illation_loss_val:.4f} | reward_ema {stats.mean_reward_ema:+.4f}"
                        f" | lookups {stats.lookups_triggered} | episodes {stats.episodes}")
            print(msg)
            history.append({
                "step": step, "lm_loss": lm_loss.item(), "illation_loss": illation_loss_val,
                "reward_ema": stats.mean_reward_ema if illation_engine else None,
                "lookups": stats.lookups_triggered if illation_engine else None,
                "episodes": stats.episodes if illation_engine else None,
                "elapsed": elapsed,
            })

        if step % eval_every == 0 or step == steps:
            val_loss = estimate_val_loss(model, val_data, SEQUENCE_LEN, device)
            print(f"  >> val_loss {val_loss:.4f} (ppl {torch.exp(torch.tensor(val_loss)).item():.2f})")
            if history and history[-1]["step"] == step:
                history[-1]["val_loss"] = val_loss
            else:
                history.append({"step": step, "val_loss": val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                model.save_pretrained(out_dir / "best")
                tok.save_pretrained(out_dir / "best")

    model.save_pretrained(out_dir / "final")
    tok.save_pretrained(out_dir / "final")
    print(f"best val_loss {best_val_loss:.4f} at step {best_step} (saved separately in {out_dir / 'best'})")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    if illation_engine is not None:
        with open(out_dir / "illation_events.txt", "w") as f:
            f.write("\n".join(stats.events[-500:]))
    print(f"saved to {out_dir}")
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "illation"], required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else HERE / "runs" / args.mode
    run(args.mode, args.steps, args.log_every, args.eval_every, out_dir)
