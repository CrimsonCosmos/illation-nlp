"""Smoke-test harden.py and analyze_internal.py against a briefly-trained InterleavedStack,
confirming the tooling itself runs correctly end-to-end (not judging real model quality --
the model here is only trained a few dozen steps from random init)."""
import random
from pathlib import Path

import torch

from tokenizer import load as load_tokenizer
from internal_model import InternalConfig
from external_model import ExternalConfig
from interleave import InterleavedStack
from vocab_schemes import InternalVocab
from harden import log_hardened_run
from analyze_internal import compressibility_check, recurrence_check, ablation_check

HERE = Path(__file__).parent


def main():
    torch.manual_seed(0)
    random.seed(0)
    tok = load_tokenizer()
    corpus_text = (HERE / "data" / "corpus.txt").read_text(encoding="utf-8")[:200_000]
    ext_ids = torch.tensor(tok.encode(corpus_text).ids, dtype=torch.long)
    ivocab = InternalVocab("ascii")

    internal_cfg = InternalConfig(vocab_size=ivocab.vocab_size, n_layer=2, n_head=4, n_kv_head=4, n_embd=64, max_seq_len=16)
    external_cfg = ExternalConfig(vocab_size=tok.get_vocab_size(), n_layer=3, n_head=4, n_kv_head=4, n_embd=96, max_seq_len=32)
    model = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    T, think_ticks = 12, 4
    for step in range(30):
        start = random.randint(0, len(ext_ids) - T - 1)
        batch = ext_ids[start : start + T].unsqueeze(0)
        loss = model.forward(batch, think_ticks=think_ticks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"trained 30 steps, final loss={loss.item():.4f}")

    # Build a hardened log over several fresh windows
    all_records = []
    for _ in range(5):
        start = random.randint(0, len(ext_ids) - T - 1)
        batch = ext_ids[start : start + T].unsqueeze(0)
        records = log_hardened_run(model, ivocab, tok, batch, think_ticks, HERE / "runs" / "v4_smoke_hardened.json")
        all_records.extend(records)
    print(f"hardened {len(all_records)} thought records, sample: {all_records[0]}")

    comp = compressibility_check([r["internal_thought"] for r in all_records], corpus_text, ivocab.base_alphabet)
    print(f"compressibility: {comp}")

    rec = recurrence_check(all_records, min_len=2)
    print(f"recurrence: {rec}")

    start = random.randint(0, len(ext_ids) - T - 1)
    batch = ext_ids[start : start + T].unsqueeze(0)
    abl = ablation_check(model, batch, think_ticks)
    print(f"ablation: {abl}")

    print()
    print("PASS (tooling ran end-to-end without error)")


if __name__ == "__main__":
    main()
