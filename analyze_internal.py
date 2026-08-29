"""Structural-validity analysis tooling (per the plan): not required for training, a
research/validation tool run against hardened internal-character logs (harden.py) to get
an empirical read on whether the internal stream is carrying real structure or is closer
to noise. Three checks:
  - compressibility: real structure compresses; noise doesn't.
  - consistency/recurrence: does the same internal pattern recur across contexts? A
    symbol/pattern needs repetition to mean anything.
  - ablation: does removing the internal model's contribution actually change External
    loss? Confirms causal relevance, not just presence.
"""
import random
import zlib
from collections import Counter

import torch


def compression_ratio(text: str) -> float:
    """compressed_bytes / raw_bytes -- lower means more structure (real text/code
    compresses to roughly 0.3-0.5x its raw size; genuinely random bytes barely compress
    below ~1.0x since there's no redundancy to exploit)."""
    raw = text.encode("utf-8", errors="replace")
    if not raw:
        return 1.0
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw)


def compressibility_check(hardened_thoughts, reference_text, alphabet):
    """hardened_thoughts: list[str]. reference_text: real prose sample, same rough length,
    for an apples-to-apples compression baseline. alphabet: the internal vocab's own base
    characters, used to generate a length-matched random-character baseline."""
    joined = "".join(hardened_thoughts)
    n = max(len(joined), 1)
    random_baseline = "".join(random.choice(alphabet) for _ in range(n))
    ref = reference_text[:n] if len(reference_text) >= n else (reference_text * (n // max(len(reference_text), 1) + 1))[:n]
    return {
        "internal_stream_ratio": compression_ratio(joined),
        "random_baseline_ratio": compression_ratio(random_baseline),
        "real_text_ratio": compression_ratio(ref),
        "n_chars": n,
    }


def recurrence_check(records, min_len=3):
    """records: list of {"context_text", "internal_thought"} from harden.log_hardened_run.
    Cheap proxy for 'does structure repeat': count how often any substring of length
    >= min_len recurs across different positions' thoughts. Real, reusable structure
    should show substrings that recur far more than a length-matched random baseline
    would by chance; report the observed recurrence rate for direct comparison against
    compressibility_check's random baseline (same underlying question, second angle)."""
    thoughts = [r["internal_thought"] for r in records if r["internal_thought"]]
    substr_counts = Counter()
    for t in thoughts:
        for i in range(len(t) - min_len + 1):
            substr_counts[t[i : i + min_len]] += 1
    recurring = {s: c for s, c in substr_counts.items() if c > 1}
    total_substrs = sum(substr_counts.values())
    recurring_substrs = sum(recurring.values())
    return {
        "n_thoughts": len(thoughts),
        "distinct_substrings": len(substr_counts),
        "recurring_substrings": len(recurring),
        "recurrence_rate": recurring_substrs / total_substrs if total_substrs else 0.0,
        "top_recurring": sorted(recurring.items(), key=lambda kv: -kv[1])[:10],
    }


@torch.no_grad()
def ablation_check(model, external_token_ids, think_ticks):
    """Compares External loss with the Internal model's contribution intact vs. zeroed
    out at every layer (ext_reads_int forced to 0), for the same input. A meaningful gap
    confirms the internal stream is doing real causal work, not sitting there unused."""
    was_training = model.training
    model.eval()

    normal_loss = model.forward(external_token_ids, think_ticks=think_ticks).item()

    orig_forwards = [type(m).forward for m in model.ext_reads_int]

    def zero_forward(self, query_x, kv_layer_states):
        return torch.zeros_like(query_x)

    for m in model.ext_reads_int:
        m.forward = zero_forward.__get__(m, type(m))
    ablated_loss = model.forward(external_token_ids, think_ticks=think_ticks).item()
    for m, orig in zip(model.ext_reads_int, orig_forwards):
        m.forward = orig.__get__(m, type(m))

    if was_training:
        model.train()

    return {
        "normal_loss": normal_loss,
        "ablated_loss": ablated_loss,
        "delta": ablated_loss - normal_loss,
        "internal_helps": ablated_loss > normal_loss,
    }
