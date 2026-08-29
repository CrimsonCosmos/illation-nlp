"""Illation: a training-only reasoning process, built on a pretrained HF causal LM.

At a sampled position, the model (using only its own current weights -- no human/synthetic
rationale data) autoregressively imagines a short "thought" about what it's just read.
While imagining, it self-monitors uncertainty (entropy of its own next-token distribution);
if uncertainty on a word it's forming is high enough, it pauses, does a real dictionary
lookup (WordNet, offline) for that word, splices the definition into its own context, and
keeps thinking. Deeper lookups within the same thought require exponentially more
uncertainty, bounding runaway chains.

The thought is then scored purely by whether having imagined it helps the model predict
what actually happens next in the real text (REINFORCE), and that signal is backpropagated
into the base weights. Nothing about a thought's content is ever a supervised target --
only whether producing it helped.

Two-phase design (standard for RL-on-LM setups, and what makes the speedup safe):
  Phase A -- imagine (no_grad, KV-cached via HF's DynamicCache, batched across rows with
    per-row graduation): fast autoregressive sampling. Nothing here needs gradients, so it
    freely uses the model's own inference-oriented cache. All rows share one context length
    (`pos`), so the shared portion of generation runs as one batched forward per step (HF's
    DynamicCache -- like nanochat's -- tracks one sequence length for the whole batch, not
    per row). The instant a row triggers a lookup it "graduates": its cache slice is copied
    out (`copy.deepcopy` + `batch_select_indices`) so it can't contaminate the rows still
    sharing the batch, and it finishes there with the same interleaved sample/lookup/continue
    behavior a lone sequence would get.
  Phase B -- score (grad, one plain forward per row, no cache): a single ordinary forward
    pass over the assembled [context, thought, future] sequence gives, from the same
    logits, both the future-token loss (does the thought help?) and the log-probs/entropy
    of exactly the tokens the policy actually sampled (for REINFORCE) -- cleanly separating
    "what did it imagine" from "how good was it, and what's the gradient".
"""
import copy
import re
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn

THOUGHT_START = "<|thought_start|>"
THOUGHT_END = "<|thought_end|>"
LOOKUP = "<|lookup|>"

WORD_RE = re.compile(r"[A-Za-z]{3,}")


@dataclass
class IllationConfig:
    max_thought_len: int = 12
    future_window: int = 16
    entropy_threshold: float = 3.0
    entropy_backoff: float = 2.0
    max_lookups_per_thought: int = 3
    max_lookup_tokens: int = 12
    reward_baseline_momentum: float = 0.95
    thought_weight: float = 0.5
    entropy_bonus: float = 0.01


@dataclass
class IllationStats:
    lookups_triggered: int = 0
    thought_tokens_generated: int = 0
    episodes: int = 0
    mean_reward_ema: float = 0.0
    events: list = field(default_factory=list)


def _entropy_nograd(logits_row: torch.Tensor) -> float:
    with torch.no_grad():
        p = F.softmax(logits_row.float(), dim=-1)
        logp = torch.log(p.clamp_min(1e-9))
        return float(-(p * logp).sum())


def _lookup_definition(word: str) -> str | None:
    synsets = wn.synsets(word.lower())
    if not synsets:
        return None
    return synsets[0].definition()


def _extract_row_cache(pkv, row: int):
    """Graduate one row out of a shared batched DynamicCache into its own single-row
    cache. Deep-copies first so the row can keep growing independently without the rest
    of the shared batch (or the original cache object) ever being touched."""
    solo = copy.deepcopy(pkv)
    solo.batch_select_indices(torch.tensor([row], device=solo.layers[0].device))
    return solo


class _RowThought:
    """Bookkeeping for one row's imagined thought. `tokens` records every token emitted
    during generation in order, tagged with whether it was policy-sampled (a real thought
    token) or injected (part of a spliced-in dictionary definition) -- only sampled tokens
    get REINFORCE credit."""

    __slots__ = ("tokens", "lookup_depth", "word_buffer", "steps_done", "done")

    def __init__(self):
        self.tokens = []          # list of (token_id, is_sampled)
        self.lookup_depth = 0
        self.word_buffer = ""
        self.steps_done = 0
        self.done = False


class IllationEngine:
    def __init__(self, model, tokenizer, cfg: IllationConfig, device, max_seq_len: int):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.device = device
        self.max_seq_len = max_seq_len  # our training window, independent of the model's own max context
        self.reward_baseline = 0.0
        self._baseline_init = False

        self.thought_start_id = tokenizer.convert_tokens_to_ids(THOUGHT_START)
        self.thought_end_id = tokenizer.convert_tokens_to_ids(THOUGHT_END)
        self.lookup_id = tokenizer.convert_tokens_to_ids(LOOKUP)

    def _encode_fragment(self, text, max_tokens):
        return self.tok.encode(text, add_special_tokens=False)[:max_tokens]

    def _update_word_buffer(self, row: _RowThought, token_id: int):
        piece = self.tok.decode([token_id])
        if piece.strip() and not piece.startswith(" "):
            row.word_buffer += piece
        else:
            row.word_buffer = piece.strip()

    def _maybe_lookup(self, row: _RowThought, ent: float, stats: IllationStats, room: int = 10**9):
        cfg = self.cfg
        gate = cfg.entropy_threshold * (cfg.entropy_backoff ** row.lookup_depth)
        if ent <= gate or row.lookup_depth >= cfg.max_lookups_per_thought:
            return None
        if room < cfg.max_lookup_tokens + 2:
            return None  # not enough room left in our training context window
        if not WORD_RE.fullmatch(row.word_buffer or ""):
            return None
        definition = _lookup_definition(row.word_buffer)
        if definition is None:
            return None
        row.lookup_depth += 1
        stats.lookups_triggered += 1
        stats.events.append(
            f"lookup[{row.lookup_depth}] '{row.word_buffer}' (H={ent:.2f} > {gate:.2f}): {definition}"
        )
        row.word_buffer = ""
        def_ids = self._encode_fragment(definition, cfg.max_lookup_tokens)
        return [self.lookup_id] + def_ids + [self.lookup_id]

    @torch.no_grad()
    def _continue_solo(self, row: _RowThought, pkv, last_logits: torch.Tensor, stats: IllationStats):
        """Finish a graduated row's remaining thought steps on its own cache. `last_logits`
        are the logits for the token right after everything currently in `pkv`."""
        cfg = self.cfg
        while row.steps_done < cfg.max_thought_len and pkv.get_seq_length() < self.max_seq_len - 2:
            probs = F.softmax(last_logits.float(), dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1))
            ent = _entropy_nograd(last_logits)

            row.tokens.append((next_id, True))
            row.steps_done += 1
            self._update_word_buffer(row, next_id)

            out = self.model(torch.tensor([[next_id]], device=self.device), past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            last_logits = out.logits[0, -1, :]

            room = self.max_seq_len - pkv.get_seq_length()
            lookup_ids = self._maybe_lookup(row, ent, stats, room)
            if lookup_ids:
                for tid in lookup_ids:
                    row.tokens.append((tid, False))
                out = self.model(torch.tensor([lookup_ids], device=self.device), past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                last_logits = out.logits[0, -1, :]
        row.done = True

    @torch.no_grad()
    def _imagine_batch(self, context: torch.Tensor, nb: int, stats: IllationStats):
        """Phase A. Returns a list of nb finished _RowThought objects."""
        cfg = self.cfg
        rows = [_RowThought() for _ in range(nb)]

        out = self.model(context, use_cache=True)  # prefill the real context
        pkv = out.past_key_values
        ts_batch = torch.full((nb, 1), self.thought_start_id, device=self.device, dtype=torch.long)
        out = self.model(ts_batch, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        last_logits = out.logits[:, -1, :]  # (nb, vocab)

        active = list(range(nb))
        for _step in range(cfg.max_thought_len):
            if not active or pkv.get_seq_length() >= self.max_seq_len - 2:
                break
            probs = F.softmax(last_logits[active].float(), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

            graduate, still_active = [], []
            room = self.max_seq_len - pkv.get_seq_length() - 1  # uniform across the shared batch
            for i, row_idx in enumerate(active):
                row = rows[row_idx]
                next_id = int(sampled[i])
                ent = _entropy_nograd(last_logits[row_idx])
                row.tokens.append((next_id, True))
                row.steps_done += 1
                self._update_word_buffer(row, next_id)
                lookup_ids = self._maybe_lookup(row, ent, stats, room)
                (graduate if lookup_ids is not None else still_active).append((row_idx, lookup_ids))

            # Advance the WHOLE shared cache with every row's actual sampled token (rows
            # not in `active` this round just get fed a harmless placeholder -- their
            # output is discarded, they've already finished).
            next_tok = torch.full((nb, 1), self.thought_start_id, device=self.device, dtype=torch.long)
            sampled_map = {row_idx: int(sampled[i]) for i, row_idx in enumerate(active)}
            for row_idx, tok_id in sampled_map.items():
                next_tok[row_idx, 0] = tok_id
            out = self.model(next_tok, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            last_logits = out.logits[:, -1, :]

            # Now graduate: extract (cache already includes this step's real token).
            for row_idx, lookup_ids in graduate:
                solo_pkv = _extract_row_cache(pkv, row_idx)
                solo_last_logits = last_logits[row_idx]
                if lookup_ids:
                    for tid in lookup_ids:
                        rows[row_idx].tokens.append((tid, False))
                    lu_out = self.model(torch.tensor([lookup_ids], device=self.device), past_key_values=solo_pkv, use_cache=True)
                    solo_pkv = lu_out.past_key_values
                    solo_last_logits = lu_out.logits[0, -1, :]
                self._continue_solo(rows[row_idx], solo_pkv, solo_last_logits, stats)

            active = [row_idx for row_idx, _ in still_active]

        for row_idx in active:
            rows[row_idx].done = True

        stats.episodes += nb
        stats.thought_tokens_generated += sum(sum(1 for _, s in r.tokens if s) for r in rows)
        return rows

    def run_batch(self, batch_ids: torch.Tensor, pos: int, stats: IllationStats):
        """batch_ids: (nb, L) real token ids for nb sequences sharing the same `pos`."""
        cfg = self.cfg
        nb = batch_ids.size(0)
        context = batch_ids[:, : pos + 1]
        future = batch_ids[:, pos + 1: pos + 1 + cfg.future_window]
        n_future = future.size(1)
        if n_future < 4:
            return None

        rows = self._imagine_batch(context, nb, stats)

        # Phase B: loss_without is identical-length across rows (shared pos + fixed
        # future window), so it batches for free with no padding needed.
        with torch.no_grad():
            without_ids = torch.cat([context, future], dim=1)
            without_logits = self.model(without_ids).logits
            without_logits = without_logits[:, -(n_future + 1):-1, :]
            without_targets = without_ids[:, -n_future:]
            loss_without_per_row = F.cross_entropy(
                without_logits.reshape(-1, without_logits.size(-1)), without_targets.reshape(-1),
                reduction="none",
            ).view(nb, n_future).mean(dim=1)

        losses = []
        for i, row in enumerate(rows):
            token_ids = [t for t, _ in row.tokens]
            sampled_flags = [s for _, s in row.tokens]
            ctx_ids = context[i].tolist()
            full_ids = ctx_ids + [self.thought_start_id] + token_ids + [self.thought_end_id] + future[i].tolist()
            thought_offset = len(ctx_ids) + 1  # index of first thought token in full_ids
            sampled_positions = [thought_offset + j for j, s in enumerate(sampled_flags) if s]

            ids_tensor = torch.tensor([full_ids[-self.max_seq_len:]], device=self.device)
            trim = len(full_ids) - ids_tensor.size(1)  # how much got truncated off the front, if any
            logits = self.model(ids_tensor).logits  # full grad, no cache

            future_logits = logits[0, -(n_future + 1):-1, :]
            future_targets = ids_tensor[0, -n_future:]
            loss_with = F.cross_entropy(future_logits, future_targets)

            log_probs, ent_terms = [], []
            for p in sampled_positions:
                pp = p - trim
                if pp < 1:
                    continue  # truncated off the front; rare (only if context is very long)
                logit_row = logits[0, pp - 1, :]
                logp = F.log_softmax(logit_row.float(), dim=-1)
                log_probs.append(logp[ids_tensor[0, pp]])
                prob = logp.exp()
                ent_terms.append(-(prob * logp).sum())

            if not log_probs:
                continue
            sum_log_prob = torch.stack(log_probs).sum()
            mean_entropy = torch.stack(ent_terms).mean()

            reward = float(loss_without_per_row[i]) - float(loss_with.detach())
            if not self._baseline_init:
                self.reward_baseline = reward
                self._baseline_init = True
            else:
                m = cfg.reward_baseline_momentum
                self.reward_baseline = m * self.reward_baseline + (1 - m) * reward
            advantage = reward - self.reward_baseline

            pg_loss = -advantage * sum_log_prob - cfg.entropy_bonus * mean_entropy
            losses.append(cfg.thought_weight * (pg_loss + loss_with))

        if not losses:
            return None
        stats.mean_reward_ema = self.reward_baseline
        return torch.stack(losses).mean()
