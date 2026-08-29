"""Phase 3: continual learning experiment. Per the user's explicit framing, continual
adaptation is stored ONLY in the Internal model + interleaving cross-attention weights --
the External model is frozen for this entire script, never updated. This is exactly
user_state.py's design (per-user Internal+interleaving state, External shared/frozen),
so this script is the real test of that mechanism, using user_state.py's own
checkpoint/history machinery rather than a parallel ad-hoc save path.

Protocol: train sequentially on distinct books (data/books/*.txt), one "session" per book,
in a fixed order. After each session, evaluate held-out loss on EVERY session seen so far
(not just the current one) -- this gives a session x session eval matrix, the standard
continual-learning readout:
  - the diagonal (eval on session i right after training session i) shows within-session
    learning
  - later rows show whether earlier sessions' loss holds steady (no forgetting), improves
    (positive transfer), or rises (catastrophic forgetting) as more sessions are trained
Each session's Internal+interleaving state is checkpointed via user_state.py under a single
user id, so `user_state.history(user_id)` is literally the book-by-book evolution of the
Internal model, and any session's state can be restored with `user_state.revert(...)`.

Gumbel-Softmax temperature is annealed linearly across the FULL cumulative step budget
(not reset per session) -- giving the internal signal more time to become less noisy as
continual training accumulates, addressing the "noisy signal is hard to learn to rely on"
risk flagged before the first controlled experiment.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from tokenizer import load as load_tokenizer
from internal_model import InternalConfig
from external_model import ExternalConfig
from interleave import InterleavedStack
from vocab_schemes import InternalVocab
import user_state

HERE = Path(__file__).parent
BOOKS_DIR = HERE / "data" / "books"
SESSION_ORDER = [
    "jekyll_and_hyde", "war_of_the_worlds", "frankenstein", "picture_of_dorian_gray",
    "sherlock_holmes", "pride_and_prejudice", "dracula", "moby_dick",
]  # smallest to largest, so early sessions are cheap sanity checks


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_phase1_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt["model"], ExternalConfig(**ckpt["config"])


def sample_batch(ids, T, batch_size, rng):
    max_start = len(ids) - T - 1
    starts = [rng.randint(0, max_start) for _ in range(batch_size)]
    return torch.stack([ids[s : s + T] for s in starts])


def gumbel_tau_at(cumulative_step, total_steps, tau_start=1.0, tau_end=0.3):
    frac = min(cumulative_step / max(total_steps, 1), 1.0)
    return tau_start + (tau_end - tau_start) * frac


def run(phase1_ckpt, vocab_scheme, think_ticks, steps_per_session, block_size=16, batch_size=4,
        internal_n_layer=3, internal_n_embd=128, gate_init=0.3, seed=0, out_dir=None,
        user_id="continual_experiment", log_every=25, eval_batches=5):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = get_device()
    print(f"device={device} vocab_scheme={vocab_scheme} think_ticks={think_ticks} gate_init={gate_init}")

    tok = load_tokenizer()

    # All 8 books, tokenized once, split train/val per book.
    sessions = {}
    full_text_for_combo_vocab = ""
    for name in SESSION_ORDER:
        text = (BOOKS_DIR / f"{name}.txt").read_text(encoding="utf-8", errors="replace")
        full_text_for_combo_vocab += text[:50_000]
        ids = torch.tensor(tok.encode(text).ids, dtype=torch.long)
        n_val = max(int(len(ids) * 0.05), block_size * 4)
        sessions[name] = {"train": ids[:-n_val], "val": ids[-n_val:], "n_tokens": len(ids)}
        print(f"  session {name}: {len(ids)} tokens ({len(sessions[name]['train'])} train / {n_val} val)")

    state_dict, external_cfg = load_phase1_checkpoint(phase1_ckpt, device)
    external_cfg.max_seq_len = max(external_cfg.max_seq_len, block_size)

    ivocab = InternalVocab(vocab_scheme, corpus_text=full_text_for_combo_vocab if vocab_scheme == "capped_combinatorial" else None)
    internal_cfg = InternalConfig(vocab_size=ivocab.vocab_size, n_layer=internal_n_layer, n_head=4, n_kv_head=4,
                                   n_embd=internal_n_embd, max_seq_len=think_ticks + 4)

    model = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4, gate_init=gate_init).to(device)
    model.external.load_state_dict(state_dict)
    print(f"loaded Phase 1 External checkpoint from {phase1_ckpt}")

    # Freeze External entirely -- continual adaptation lives ONLY in Internal + interleaving.
    # Gradient still flows correctly BACK THROUGH External's frozen forward computation to
    # reach Internal/cross-attn params (autograd tracks d(output)/d(input) through a layer
    # regardless of whether that layer's own weights require grad -- this is the same
    # mechanism standard frozen-backbone/adapter-tuning setups rely on), it's just that
    # External's own weights never receive an optimizer step.
    for p in model.external.parameters():
        p.requires_grad = False

    trainable_params = (
        list(model.internal.parameters())
        + list(model.ext_reads_int.parameters())
        + list(model.int_reads_ext.parameters())
        + [model.thought_start]
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.external.parameters())
    print(f"trainable (Internal+interleaving): {n_trainable/1e6:.2f}M | frozen (External): {n_frozen/1e6:.2f}M")

    opt = torch.optim.AdamW(trainable_params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)

    out_dir = Path(out_dir) if out_dir else HERE / "runs" / "v4_phase3_continual"
    out_dir.mkdir(parents=True, exist_ok=True)

    user_state.init_user(user_id, model)

    total_steps = steps_per_session * len(SESSION_ORDER)
    cumulative_step = 0
    eval_matrix = {}  # {session_index_trained_through: {session_name: val_loss}}
    session_history = []
    t0 = time.time()

    @torch.no_grad()
    def eval_session(name):
        model.eval()
        losses = [
            model.forward(sample_batch(sessions[name]["val"], block_size, batch_size, rng).to(device),
                          think_ticks=think_ticks).item()
            for _ in range(eval_batches)
        ]
        model.train()
        return sum(losses) / len(losses)

    for session_idx, name in enumerate(SESSION_ORDER):
        print(f"\n=== session {session_idx + 1}/{len(SESSION_ORDER)}: {name} ===")
        train_ids = sessions[name]["train"]
        step_log = []
        for step in range(1, steps_per_session + 1):
            cumulative_step += 1
            model.internal.config.gumbel_tau = gumbel_tau_at(cumulative_step, total_steps)
            batch = sample_batch(train_ids, block_size, batch_size, rng).to(device)
            loss = model.forward(batch, think_ticks=think_ticks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            if step % log_every == 0 or step == 1:
                elapsed = time.time() - t0
                print(f"  step {step:4d}/{steps_per_session} | loss {loss.item():.4f} | "
                      f"tau {model.internal.config.gumbel_tau:.3f} | elapsed {elapsed:7.1f}s")
                step_log.append({"cumulative_step": cumulative_step, "loss": loss.item()})

        # Evaluate on every session seen so far.
        row = {}
        for prior_idx in range(session_idx + 1):
            prior_name = SESSION_ORDER[prior_idx]
            row[prior_name] = eval_session(prior_name)
        eval_matrix[name] = row
        print(f"  eval after {name}: {row}")

        # Checkpoint via user_state.py -- this IS the continual-learning mechanism's
        # actual persistence, not a separate ad-hoc save.
        commit = user_state.save_checkpoint(user_id, model, label=f"after {name}")
        print(f"  checkpointed: {commit[:8]}")

        session_history.append({"session_idx": session_idx, "session": name, "steps": step_log, "eval": row, "commit": commit})

    with open(out_dir / "session_history.json", "w") as f:
        json.dump(session_history, f, indent=2)
    with open(out_dir / "eval_matrix.json", "w") as f:
        json.dump(eval_matrix, f, indent=2)

    print(f"\nsaved to {out_dir}")
    print(f"user_state history for {user_id}:")
    for h, msg in user_state.history(user_id):
        print(f"  {h[:8]} {msg}")

    return session_history, eval_matrix


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-ckpt", type=str, required=True)
    ap.add_argument("--vocab-scheme", choices=["ascii", "curated_unicode", "capped_combinatorial"], default="ascii")
    ap.add_argument("--think-ticks", type=int, default=8)
    ap.add_argument("--steps-per-session", type=int, default=250)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--gate-init", type=float, default=0.3)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--user-id", type=str, default="continual_experiment")
    args = ap.parse_args()
    run(args.phase1_ckpt, args.vocab_scheme, args.think_ticks, args.steps_per_session, args.block_size,
        gate_init=args.gate_init, out_dir=args.out, user_id=args.user_id)
