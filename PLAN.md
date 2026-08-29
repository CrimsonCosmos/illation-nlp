# Illation v4: Two-Model, Per-Layer-Interleaved Architecture

## Context

Illation is a training-only mechanism meant to give a language model extra "imagining" capacity while it reads, the way a person forms mental images while reading a novel — the goal being data efficiency (more understanding extracted from less text), not inference-time reasoning. Three prior versions (custom GPT → nanochat GPT → HF SmolLM2, all in `illation-nlp/`) used a single shared model that discretely sampled "thought tokens" mid-sequence and graded them with REINFORCE against the resulting future-token loss. That design is now explicitly rejected: the user does not want the *content* of the model's internal thinking to ever be reward-signaled or otherwise told what to think — only the real output should be graded, the same way a person's literal internal monologue isn't "graded," only what they actually say.

The replacement design, arrived at over an extended back-and-forth, is two separate models built on nanochat's codebase (`illation-nlp/nanochat_ref/nanochat/gpt.py`):

- An **Internal model**: character-level tokenized (every character its own token, ~256-token vocab), so no English/word-level prior is imposed — an internal "thought language" is meant to emerge purely from training pressure. It sees everything the External model sees, and more (per the user: "ideally it sees everything... even more").
- An **External/Outer model**: a standard nanochat GPT, BPE-tokenized, the one whose output is the model's real, graded speech.

The two are **interleaved at every layer**, not connected at a single hand-off point — this was explicitly and repeatedly confirmed after two rejected simpler designs (recurrent-depth self-looping, and single-point "extra tokens appended to context").

Critically, only the External model's ordinary next-token loss ever exists as a training signal for content. The Internal model's parameters are shaped exclusively as an indirect consequence of backprop flowing through the connection into it — never a separate reward on what it "says."

**Revision from the original premise**: thinking duration is being made a knob the user can set at generation time, which means illation is no longer training-only — the Internal model and the interleaving mechanism now run at inference too, as a permanent part of the External model's forward pass. This is a deliberate, confirmed reversal of the project's original "illation exists only during training" constraint.

## Reconciling open questions (resolved this session)

- **Outer model source**: pretrain from scratch on nanochat's own architecture (not fine-tune a pretrained HF checkpoint like Qwen3.5-4B). This is required for true per-layer interleaving — we need to own and modify the External model's internals layer-by-layer, which isn't practical against a fixed `transformers` implementation.
- **Corpus strategy (two phases)**: ~10M tokens across our own corpus (8 existing books + Suiko's 60, `~/repos/Suiko/data/library/texts/*.txt`) is far too little to pretrain a fluent model from scratch (even at nanochat's own relaxed 12:1 token:param ratio, that's under 1M params' worth of data). So:
  - **Phase 1 (base competency)**: plain nanochat pretraining on a small ClimbMix shard (illation mechanism off) — enough tokens for a genuinely fluent small model (target ~50-100M params, sized for one A10G; ~600M-2B ClimbMix tokens, i.e. ~10-30 of nanochat's ~61M-token shards, from `nanochat/dataset.py`'s existing download logic).
  - **Phase 2 (illation)**: continue training that fluent base on our own curated corpus (raw `.txt` files, concatenated the same way as the existing `data/corpus.txt` pipeline) with the two-model illation mechanism switched on. This is where "imagining while reading" actually happens.

## Architecture

### Internal model (`illation-nlp/internal_model.py`, new)
- nanochat `GPTConfig`-style, but tiny. **Vocabulary size is an empirical hyperparameter, not a fixed upfront choice** — the user wants whichever alphabet size actually produces the best internal thinking, not whichever sounds most principled in the abstract. `internal_model.py` takes vocab scheme as a config, supporting a small family of candidate schemes so they can be compared head-to-head (see "Vocab sweep" under Verification below):
  1. Plain ASCII (~100 tokens) — the minimal-prior baseline.
  2. Curated Unicode (~2,000-5,000 tokens) — Latin extended + common technical/math/programming symbols, no combinatorial scheme.
  3. Full Unicode + capped combinatorial (as previously discussed: full code-point alphabet, plus an escape-triggered combination scheme bounded so combos stay frequent enough to be learnable).
- `n_embd` scales with whichever vocab wins. `n_layer` is **decoupled from the External model's `n_layer`** — see "Interleaving mechanism" below for why a strict 1:1 depth match isn't required or even desirable.
- Runs continuously alongside the External model's forward pass (both during training and, per the revision above, during inference too) — driven by the interleaving mechanism below, not by its own independent generation loop.
- Uses reused pieces of `nanochat_ref/nanochat/gpt.py`: `CausalSelfAttention`, `MLP`, `Block`, rotary embeddings, `norm()`. Character-level tokenizer is new (trivial fixed-vocab mapping, no BPE needed).

### External model (`illation-nlp/external_model.py`, new — thin wrapper around a modified `GPT`)
- Standard nanochat `GPT` (reuse `nanochat_ref/nanochat/gpt.py` directly, or a local fork if the interleaving hook can't be added non-invasively — see below), BPE-tokenized via nanochat's own tokenizer path.
- Sized to fit a single A10G (g5.xlarge): the existing AWS instance `i-05537902fef7cd49f` (currently STOPPED) is already the right box for this.

### Interleaving mechanism (`illation-nlp/interleave.py`, new)

**Why not a strict 1:1 layer pairing.** The original draft required the Internal model's `n_layer` to exactly match the External model's, so "layer `i`" on each side could be paired directly. That's an artificial constraint, not a real requirement — it forces the Internal model to be exactly as deep as the External model even though there's no reason the two need the same capacity, and it makes the two architectures needlessly coupled (can't resize one without resizing the other).

Instead: the Internal model runs its own full stack (its own independent `n_layer`) once per "tick" (see thinking-duration below), producing a set of per-layer states from that tick. Every External layer `i` then reads from the Internal model via **attention-pooling across all of the Internal model's layer states** (a small learned cross-attention where the External layer's residual stream provides the query, and the stacked Internal per-layer states — from the current tick — provide keys/values). This means:
- Every External layer still genuinely participates in reading from the Internal model (satisfying "every layer, not one hand-off point").
- The two models' depths are fully decoupled — either can be resized independently.
- Early vs. late External layers can naturally learn to attend to early vs. late (more "raw" vs. more "processed") Internal states, which a forced 1:1 pairing would have prevented from being a learned choice at all.

The mirror direction — **Internal reads External** — works the same way: at the start of each tick, the Internal model's first layer takes External's current full per-layer state stack as additional cross-attention input (queries from Internal, keys/values from all External layers), satisfying "it sees everything the external model knows... and even more."

This is implemented as a new `InterleavedStack` module owning one External `GPT`-style stack, one Internal `GPT`-style stack, and the cross-attention pooling in both directions — not by patching nanochat's `Block`/`GPT` classes in place (keeps `nanochat_ref/` untouched and reusable as-is for the Phase-1 base-pretraining run, which doesn't need any of this).

### Differentiability: no reward on content

The Internal model operates on discrete characters, which would normally block gradient flow — picking a specific character (like flipping a coin and landing on one face) is a hard, non-differentiable choice, so gradients can't flow back through it to say "make this choice slightly more/less likely."

**Gumbel-Softmax with a straight-through estimator**, in plain terms: instead of the Internal model committing to one hard character at each step, during training it produces a *soft blend* over all possible characters — e.g. 70% weight on 't', 20% on 'h', small amounts spread across the rest — and it's that blend (a real, continuous number) that actually gets fed forward, both as the Internal model's own next-step input and across the cross-attention coupling into the External model. Because it's a continuous blend rather than a coin-flip, ordinary backprop can flow through it just like any other number in the network. "Straight-through" means: in the forward pass we still snap the blend to its hard winning character for anything that needs an actual discrete symbol (e.g. the hardening/inspection tool, or computing what the "real" internal sequence was), but for the backward pass we pretend the soft blend was used the whole time, so gradients still flow normally. Net effect: the model *behaves* like it's making discrete character choices, but training can still shape it with plain backprop — no reward signal, no REINFORCE, no separate grading of content anywhere in this path. The only gradient signal touching Internal model parameters is backprop from the External model's real next-token loss, matching the user's explicit requirement.

### Thinking duration (`illation-nlp/halting.py`, new)

**v1: fixed tick count, not learned.** Before each External-model output token, the Internal model runs for a configurable, fixed number of "ticks" (each tick = one pass through the interleaved stack described above) — e.g. `think_ticks=8` — set as a plain config value during training and, per the confirmed reversal above, as a **generation-time parameter** at inference (e.g. `generate.py --think-ticks 16` to make the model "think harder" on a given prompt, or `--think-ticks 1` to make it fast/shallow).

This deliberately skips a fully learned ACT-style halting head for v1. A learned stop-probability head trained via REINFORCE on an efficiency-only reward has a well-known failure mode: if the minimum floor is small (e.g. 2 ticks) and the model hasn't yet learned that more thinking actually helps, the efficiency pressure can make it collapse straight to the floor and never explore using more ticks — silently making the entire Internal model useless despite the architecture being "correct." A fixed, manually-set tick count sidesteps that risk entirely, is trivially verifiable (you can just try different values and watch loss/output quality), and doubles as exactly the generation-time knob being asked for.

**Stretch goal (post-v1, not in this implementation pass): learned adaptive halting.** Once the fixed-tick version is proven to actually help (Internal model demonstrably reduces External loss at a given tick count), revisit a learned per-token halting decision — likely with mitigations for the collapse failure mode (e.g. annealing the efficiency penalty in slowly, only after a warmup period where thinking is "free," so the model has a chance to discover thinking helps before being pressured to economize). This stays out of scope for the initial build.

### Hardening / inspection tooling (`illation-nlp/harden.py`, new)
A separate, no-grad utility (not on any training path) that takes the logged soft Gumbel-Softmax distributions from a forward pass and argmaxes them into an actual character string per tick, for logging and manual inspection — explicitly requested as an important deliverable ("we need this"), independent of whether the result looks like real language.

### Structural-validity analysis tooling (`illation-nlp/analyze_internal.py`, new)
Since the user is fine with the internal stream being uninterpretable *to us* as long as it's actually doing structured work, add lightweight analysis (not required for training, a research/validation tool) that can be run against hardened internal-character logs:
- **Compressibility check**: compression ratio of the hardened internal stream vs. real English text vs. genuinely random characters (via stdlib `zlib`) — real structure compresses; noise doesn't.
- **Consistency/recurrence check**: do similar External-model contexts (e.g. by embedding similarity) tend to produce similar/recurring internal character patterns?
- **Ablation check**: zero out or shuffle the internal cross-attention contribution at inference-adjacent eval time and measure the resulting change in External-model loss/output — confirms the internal stream is doing real causal work, not decoration.

### Per-user persistent online adaptation (`illation-nlp/user_state.py`, new)

Beyond the offline Phase 1/2 training above, the Internal model keeps learning during real usage, per-user, indefinitely — not reset each session. Mechanically:

- **The External model stays frozen and shared** across every user — it's the large, expensive, stable "speaking" component. Only the **Internal model + interleaving cross-attention weights** (much smaller) get a separate copy per user; the architecture (vocab, `n_layer`, `n_embd`, etc.) must stay identical across users so weight files remain interchangeable — see "weight sharing" below.
- During live usage, once a turn's actual continuation is known (the model's own generated tokens, or the user's next message), that's a real observed continuation — a genuine self-supervised next-token loss can be computed retroactively and backpropagated through the same Gumbel-Softmax/straight-through path used in training, taking a small optimizer step against that user's Internal-model + interleaving weights only. This keeps the "no reward on Internal content" property intact — it's the same ordinary-backprop-through-External-loss mechanism as offline training, just running continuously instead of stopping after Phase 2.
- **Coarse, git-backed checkpointing**: rather than committing after every turn (binary-blob bloat risk in git), the user's current Internal+interleaving weight file is committed at a checkpoint interval (e.g. every N sessions, or an explicit manual "save" point) to a dedicated state-history git repo — `user_states/<user_id>/internal_weights.safetensors`, one commit per checkpoint. Reverting to any prior point in that user's history is a plain `git checkout <commit> -- user_states/<user_id>/`.
- **Weight sharing ("plug and play")**: since a user's Internal+interleaving state is just a portable weight file under a fixed, shared architecture, one user can hand theirs to another — implemented as `user_state.py --import <target_user> --from <source_user_or_path>`, which copies the file into the target user's directory and creates a new checkpoint commit (so it shows up in their git history like any other checkpoint). Because it's a normal commit rather than a destructive overwrite, the target user's own prior states remain in their git history — they can revert back to their own weights at any time even after importing someone else's.

## Training pipeline changes

- `illation-nlp/train_v4.py` (new, replaces the current `train.py`'s role for this architecture — old `train.py`/`illation.py`/SmolLM2 path stays as-is/untouched, this is a fresh v4 script):
  - **Phase 1**: instantiate External model alone (Internal model + interleaving not yet active), plain nanochat pretraining loop against a downloaded ClimbMix shard slice (reuse `nanochat_ref/nanochat/dataset.py` download logic), using nanochat's own `setup_optimizer()`/Muon schedule (already ported once before in this project's history — the LR-warmup/constant/warmdown + cosine weight-decay logic from the abandoned nanochat-GPT phase should be reused, not re-derived).
  - **Phase 2**: load the Phase-1 External checkpoint, attach a freshly-initialized Internal model + `InterleavedStack` + fixed-tick-count thinking (`think_ticks` config value, no halting head in v1), continue training on the own-corpus data (8 books + Suiko's 60, concatenated the same way as the existing `data/corpus.txt`/`fetch_corpus.py` pattern) with illation active: External next-token loss backprops through the whole interleaved stack, including into the Internal model, via the Gumbel-Softmax/straight-through path — no separate loss term needed in v1 since there's no learned halting decision to grade.
- `illation-nlp/generate.py`: needs a v4-aware path. Per the confirmed reversal, illation now runs at inference too — add a `--think-ticks` CLI argument controlling how many Internal-model ticks run before each External output token (same fixed-tick mechanism as training, just settable per generation call).

## Verification plan

- Unit-level: a small script instantiating `InterleavedStack` with tiny dims and confirming shapes flow correctly and gradients reach both the External and Internal parameter sets from a single External-loss `.backward()` call (checked via `param.grad is not None`) — this is the core correctness property (no reward touches Internal content) and should be checked before any real training run.
- Phase 1 smoke test: short run (a few hundred steps) on a single ClimbMix shard slice, confirm loss decreases normally (same pattern used to catch the earlier catastrophic-overfitting bug via `history.json` inspection).
- Phase 2 smoke test: short run on the own corpus with illation active, confirm (a) External loss decreases, (b) hardened internal-character logs are non-degenerate (not constant/empty) at the chosen `think_ticks`, (c) a quick sweep of `think_ticks` values (e.g. 1, 4, 8, 16) shows the higher-tick settings actually help External loss at least somewhat — the basic sanity check that the Internal model is doing anything useful at all before investing in the learned-halting stretch goal.
- **Vocab sweep**: run the Phase 2 smoke test independently for each of the 3 candidate vocab schemes above (same corpus, same `think_ticks`, same step budget, everything else held constant), and compare on: (a) final External loss/perplexity — the primary signal, since it directly measures whether the Internal model is actually helping the real output; (b) the `analyze_internal.py` compressibility check — does the hardened internal stream show real structure or does it look closer to noise; (c) the consistency/recurrence check — does the scheme actually produce reusable patterns, or do larger vocabs just spread usage too thin to ever repeat. Whichever scheme wins on External loss (with the structural checks as a tiebreaker/sanity check, not the primary criterion) is what Phase 1's full run and beyond use. This directly answers "which alphabet size leads to the best internal thinking" with a measurement instead of an upfront guess.
- Run the `analyze_internal.py` compressibility + ablation checks once Phase 2 training has run long enough to be meaningful, to get an early read on whether the internal stream is carrying real structure.
- `user_state.py` checkpoint/revert/import round-trip: create two dummy per-user weight states, checkpoint both, import one into the other, confirm the target's git history still contains its own pre-import commit and `git checkout` back to it restores the original weights byte-for-byte.
- Reuse the existing AWS `g5.xlarge` box (`i-05537902fef7cd49f`) for anything beyond a local smoke test, per the user's earlier hardware upgrade.
