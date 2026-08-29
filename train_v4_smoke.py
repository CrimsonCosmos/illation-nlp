"""Phase 2 smoke test (per the approved plan's Verification section): a short, small-scale
training run of the full InterleavedStack against the real corpus, using the already-
trained BPE tokenizer (tokenizer.py/tokenizer_files/) for the External side. Confirms:
  (a) External loss actually decreases over a short run,
  (b) hardened internal-character traces are non-degenerate (not constant/empty),
  (c) a quick think_ticks sweep shows higher-tick settings help at least somewhat.

This intentionally skips Phase 1 (ClimbMix base-competency pretraining) -- that requires
real GPU-scale compute (see the plan's AWS section) and isn't needed to validate the
mechanism itself, which is what this script checks. The External stack here starts from
random init, not a pretrained base; we're checking "does the Internal model help reduce
loss on top of whatever the External model is learning," not final model quality.
"""
import random
import time
from pathlib import Path

import torch

from tokenizer import load as load_tokenizer
from internal_model import InternalConfig
from external_model import ExternalConfig
from interleave import InterleavedStack
from vocab_schemes import InternalVocab

HERE = Path(__file__).parent


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_corpus_ids(tok, n_chars=200_000):
    text = (HERE / "data" / "corpus.txt").read_text(encoding="utf-8")[:n_chars]
    ids = tok.encode(text).ids
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(ids, T, batch_size):
    max_start = len(ids) - T - 1
    starts = [random.randint(0, max_start) for _ in range(batch_size)]
    return torch.stack([ids[s : s + T] for s in starts])


def run(think_ticks, steps, seed=0, log_every=10):
    torch.manual_seed(seed)
    random.seed(seed)
    device = get_device()
    print(f"device={device} think_ticks={think_ticks}")

    tok = load_tokenizer()
    corpus_text = (HERE / "data" / "corpus.txt").read_text(encoding="utf-8")[:200_000]
    ext_ids = torch.tensor(tok.encode(corpus_text).ids, dtype=torch.long)

    ivocab = InternalVocab("ascii")

    internal_cfg = InternalConfig(vocab_size=ivocab.vocab_size, n_layer=2, n_head=4, n_kv_head=4, n_embd=64, max_seq_len=16)
    external_cfg = ExternalConfig(vocab_size=tok.get_vocab_size(), n_layer=3, n_head=4, n_kv_head=4, n_embd=96, max_seq_len=32)
    model = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M (internal {sum(p.numel() for p in model.internal.parameters())/1e3:.0f}K, "
          f"external {sum(p.numel() for p in model.external.parameters())/1e3:.0f}K)")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

    T = 12  # short context: O(T^2) prototype, keep small for a smoke test
    batch_size = 4
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = sample_batch(ext_ids, T, batch_size).to(device)
        # process each row separately (InterleavedStack.forward assumes a shared batch
        # marches through positions together; B>1 works fine as written, batch here for
        # gradient averaging across independent windows)
        loss = model.forward(batch, think_ticks=think_ticks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  step {step:4d} | loss {loss.item():.4f} | elapsed {elapsed:6.1f}s")

    # Hardened internal trace sample, for non-degeneracy check
    model.eval()
    with torch.no_grad():
        sample = sample_batch(ext_ids, T, 1).to(device)
        _, traces = model.forward(sample, think_ticks=think_ticks, return_traces=True)
    model.train()
    decoded = [ivocab.decode(t) for t in traces]

    return losses, decoded


if __name__ == "__main__":
    print("=== think_ticks sweep (short runs) ===")
    results = {}
    for tt in [1, 4, 8]:
        losses, decoded = run(think_ticks=tt, steps=60, seed=0)
        first10_avg = sum(losses[:10]) / 10
        last10_avg = sum(losses[-10:]) / 10
        results[tt] = (first10_avg, last10_avg, decoded)
        print(f"think_ticks={tt}: first-10 avg loss={first10_avg:.4f}  last-10 avg loss={last10_avg:.4f}")
        print(f"  sample internal traces (hardened, one per output position): {decoded}")
        print()

    print("=== summary ===")
    for tt, (first, last, decoded) in results.items():
        non_degenerate = len(set("".join(decoded))) > 1
        print(f"think_ticks={tt}: loss {first:.4f} -> {last:.4f} (Δ={first-last:+.4f})  non_degenerate_traces={non_degenerate}")
