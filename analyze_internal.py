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


MIN_RELIABLE_CHARS = 8192  # below this, zlib's fixed per-stream overhead dominates the
# ratio and swamps any real signal -- flag rather than silently report a noisy number.


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
        "reliable": n >= MIN_RELIABLE_CHARS,
    }


def recurrence_check(records, vocab_size, min_len=2):
    """records: list of {"context_text", "internal_thought"} from harden.log_hardened_run.
    Counts how often any substring of length min_len recurs across different positions'
    thoughts, AND compares it against what pure chance would produce for this vocab size
    and sample count -- raw recurrence counts are dominated by vocab-size combinatorics
    (birthday-paradox effect: for V possible substrings and n samples, chance collisions
    scale as ~n^2/(2*V^min_len)), so a larger vocab shows fewer recurrences even with zero
    real structure. signal_ratio = observed / expected_under_chance is the number that
    actually reflects whether structure is present: ~1 means indistinguishable from noise,
    >>1 means real recurring structure beyond what vocab-size math alone would predict."""
    thoughts = [r["internal_thought"] for r in records if r["internal_thought"]]
    substr_counts = Counter()
    n_substrs = 0
    for t in thoughts:
        for i in range(len(t) - min_len + 1):
            substr_counts[t[i : i + min_len]] += 1
            n_substrs += 1
    recurring = {s: c for s, c in substr_counts.items() if c > 1}
    observed_recurring = sum(recurring.values())

    possible_substrings = vocab_size ** min_len
    expected_recurring_under_chance = (n_substrs ** 2) / (2 * possible_substrings) if possible_substrings > 0 else 0.0
    if expected_recurring_under_chance > 1e-9:
        signal_ratio = observed_recurring / expected_recurring_under_chance
    else:
        signal_ratio = float("inf") if observed_recurring > 0 else 1.0

    return {
        "n_thoughts": len(thoughts),
        "n_substrings_sampled": n_substrs,
        "vocab_size": vocab_size,
        "distinct_substrings": len(substr_counts),
        "observed_recurring": observed_recurring,
        "expected_recurring_under_chance": round(expected_recurring_under_chance, 3),
        "signal_ratio": signal_ratio,
        "top_recurring": sorted(recurring.items(), key=lambda kv: -kv[1])[:10],
    }


@torch.no_grad()
def ablation_check(model, sample_fn, think_ticks, n_batches=10):
    """Compares External loss with the Internal model's contribution intact vs. zeroed
    out at every layer (ext_reads_int forced to 0), averaged over n_batches fresh samples
    from sample_fn() -- a single batch is not a measurement, it's noise (the original
    version of this check used one batch of size 1 and the resulting deltas were within
    noise of zero regardless of sign). A meaningful, consistent gap across many batches is
    what would confirm the internal stream is doing real causal work, not sitting unused."""
    was_training = model.training
    model.eval()

    orig_forwards = [type(m).forward for m in model.ext_reads_int]

    def zero_forward(self, query_x, kv_layer_states):
        return torch.zeros_like(query_x)

    normal_losses, ablated_losses = [], []
    for _ in range(n_batches):
        batch = sample_fn()
        normal_losses.append(model.forward(batch, think_ticks=think_ticks).item())
        for m in model.ext_reads_int:
            m.forward = zero_forward.__get__(m, type(m))
        ablated_losses.append(model.forward(batch, think_ticks=think_ticks).item())
        for m, orig in zip(model.ext_reads_int, orig_forwards):
            m.forward = orig.__get__(m, type(m))

    if was_training:
        model.train()

    normal_mean = sum(normal_losses) / len(normal_losses)
    ablated_mean = sum(ablated_losses) / len(ablated_losses)
    deltas = [a - n for a, n in zip(ablated_losses, normal_losses)]
    delta_mean = sum(deltas) / len(deltas)
    delta_std = (sum((d - delta_mean) ** 2 for d in deltas) / len(deltas)) ** 0.5

    return {
        "n_batches": n_batches,
        "normal_loss_mean": normal_mean,
        "ablated_loss_mean": ablated_mean,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        # a delta consistently larger than its own spread across batches is the signal
        # that matters -- not just whether the mean happens to be positive.
        "delta_exceeds_noise": abs(delta_mean) > delta_std,
        "internal_helps": delta_mean > 0,
    }
