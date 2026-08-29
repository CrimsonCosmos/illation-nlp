"""Hardening / inspection tooling (per the plan): a separate, no-grad utility that decodes
an InterleavedStack forward pass's internal character choices into actual text, for
logging and manual inspection. Not on any training path -- purely for looking at what the
Internal model is doing.
"""
import json
from pathlib import Path

import torch

from vocab_schemes import InternalVocab


@torch.no_grad()
def harden_batch(model, ivocab: InternalVocab, external_token_ids, think_ticks):
    """Runs one forward pass in eval mode with trace collection, returns a list (one per
    batch row... note: current InterleavedStack.forward's trace collection is row-0-only,
    see interleave.py -- for B>1 hardening, call this once per row) of per-position
    hardened strings, one string per External output position, each representing that
    position's full internal 'thought' (one character per tick)."""
    was_training = model.training
    model.eval()
    _, traces = model.forward(external_token_ids, think_ticks=think_ticks, return_traces=True)
    if was_training:
        model.train()
    return [ivocab.decode(t) for t in traces]


def log_hardened_run(model, ivocab, tokenizer, external_token_ids, think_ticks, out_path: Path):
    """Writes a JSON log: for each output position, the preceding External context (decoded
    text) and the hardened internal thought that preceded that prediction. Useful as the
    raw input to analyze_internal.py's structural checks."""
    decoded_thoughts = harden_batch(model, ivocab, external_token_ids, think_ticks)
    ids = external_token_ids[0].tolist()
    records = []
    for pos, thought in enumerate(decoded_thoughts):
        context_ids = ids[: pos + 1]
        context_text = tokenizer.decode(context_ids)
        target_id = ids[pos + 1] if pos + 1 < len(ids) else None
        records.append({
            "position": pos,
            "context_text": context_text,
            "internal_thought": thought,
            "target_token_id": target_id,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return records
