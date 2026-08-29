"""Phase 2 (illation): loads a Phase 1 External checkpoint into InterleavedStack.external,
attaches a freshly-initialized Internal model + cross-attention couplings, and continues
training on our own curated corpus with illation active. Per the plan, this is also where
the vocab sweep (which InternalVocab scheme actually helps most) happens.

Uses the same O(T^2)-per-position correctness-first InterleavedStack.forward as the smoke
test (train_v4_smoke.py) -- KV-caching is a documented follow-up, not yet built. Keep
block_size modest here accordingly.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from tokenizer import load as load_tokenizer
from internal_model import InternalConfig
from external_model import ExternalConfig, ExternalStack
from interleave import InterleavedStack
from vocab_schemes import InternalVocab
from harden import log_hardened_run
from analyze_internal import compressibility_check, recurrence_check, ablation_check

HERE = Path(__file__).parent


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_phase1_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    config = ExternalConfig(**ckpt["config"])
    return ckpt["model"], config


def sample_batch(ids, T, batch_size):
    max_start = len(ids) - T - 1
    starts = [random.randint(0, max_start) for _ in range(batch_size)]
    return torch.stack([ids[s : s + T] for s in starts])


def run(phase1_ckpt, corpus_path, vocab_scheme, think_ticks, steps, block_size=16, batch_size=4,
        internal_n_layer=3, internal_n_embd=128, out_dir=None, seed=0, log_every=20, eval_every=100):
    torch.manual_seed(seed)
    random.seed(seed)
    device = get_device()
    print(f"device={device} vocab_scheme={vocab_scheme} think_ticks={think_ticks}")

    tok = load_tokenizer()
    corpus_text = Path(corpus_path).read_text(encoding="utf-8")
    ext_ids = torch.tensor(tok.encode(corpus_text).ids, dtype=torch.long)
    n_val = max(int(len(ext_ids) * 0.02), block_size * 2)
    train_ids, val_ids = ext_ids[:-n_val], ext_ids[-n_val:]

    state_dict, external_cfg = load_phase1_checkpoint(phase1_ckpt, device) if phase1_ckpt else (None, None)
    if external_cfg is None:
        external_cfg = ExternalConfig(vocab_size=tok.get_vocab_size(), n_layer=6, n_head=6, n_kv_head=6, n_embd=384, max_seq_len=block_size)
    else:
        external_cfg.max_seq_len = max(external_cfg.max_seq_len, block_size)

    ivocab = InternalVocab(vocab_scheme, corpus_text=corpus_text if vocab_scheme == "capped_combinatorial" else None)
    internal_cfg = InternalConfig(vocab_size=ivocab.vocab_size, n_layer=internal_n_layer, n_head=4, n_kv_head=4,
                                   n_embd=internal_n_embd, max_seq_len=think_ticks + 4)

    model = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4).to(device)
    if state_dict is not None:
        model.external.load_state_dict(state_dict)
        print(f"loaded Phase 1 External checkpoint from {phase1_ckpt}")
    else:
        print("no Phase 1 checkpoint given -- External starts from random init")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M | internal vocab_size={ivocab.vocab_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)

    out_dir = Path(out_dir) if out_dir else HERE / "runs" / f"v4_phase2_{vocab_scheme}_tt{think_ticks}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()

    for step in range(1, steps + 1):
        batch = sample_batch(train_ids, block_size, batch_size).to(device)
        loss = model.forward(batch, think_ticks=think_ticks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:5d}/{steps} | loss {loss.item():.4f} | elapsed {elapsed:7.1f}s")
            history.append({"step": step, "train_loss": loss.item(), "elapsed": elapsed})

        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_losses = [model.forward(sample_batch(val_ids, block_size, batch_size).to(device), think_ticks=think_ticks).item()
                              for _ in range(10)]
            model.train()
            val_loss = sum(val_losses) / len(val_losses)
            print(f"  >> val_loss {val_loss:.4f}")
            if history and history[-1]["step"] == step:
                history[-1]["val_loss"] = val_loss
            else:
                history.append({"step": step, "val_loss": val_loss})

    torch.save({
        "external": model.external.state_dict(),
        "internal": model.internal.state_dict(),
        "ext_reads_int": model.ext_reads_int.state_dict(),
        "int_reads_ext": model.int_reads_ext.state_dict(),
        "thought_start": model.thought_start,
        "internal_cfg": internal_cfg.__dict__,
        "external_cfg": external_cfg.__dict__,
        "vocab_scheme": vocab_scheme,
        "think_ticks": think_ticks,
    }, out_dir / "final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Structural analysis pass
    model.eval()
    with torch.no_grad():
        all_records = []
        for _ in range(8):
            batch = sample_batch(val_ids, block_size, 1).to(device)
            records = log_hardened_run(model, ivocab, tok, batch, think_ticks, out_dir / "hardened_log.json")
            all_records.extend(records)
        comp = compressibility_check([r["internal_thought"] for r in all_records], corpus_text, ivocab.base_alphabet)
        rec = recurrence_check(all_records, min_len=2)
        batch = sample_batch(val_ids, block_size, 1).to(device)
        abl = ablation_check(model, batch, think_ticks)
    model.train()

    analysis = {"compressibility": comp, "recurrence": rec, "ablation": abl}
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"analysis: {analysis}")
    print(f"saved to {out_dir}")
    return history, analysis


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-ckpt", type=str, default=None)
    ap.add_argument("--corpus", type=str, default=str(HERE / "data" / "corpus.txt"))
    ap.add_argument("--vocab-scheme", choices=["ascii", "curated_unicode", "capped_combinatorial"], default="ascii")
    ap.add_argument("--think-ticks", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run(args.phase1_ckpt, args.corpus, args.vocab_scheme, args.think_ticks, args.steps, args.block_size, out_dir=args.out)
