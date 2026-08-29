"""Per-user persistent online adaptation state (per the approved plan). Only the Internal
model + interleaving cross-attention weights are per-user -- the External model stays
frozen and shared across everyone. State lives in its own git repo (user_states/, git-
init'd separately from the code repo -- see .gitignore) so checkpoint history and revert
are just normal git operations, kept out of the code repo's own history.

Coarse checkpointing: callers decide when to call save_checkpoint (e.g. every N sessions,
or an explicit "save" point) -- this module doesn't auto-commit on every turn, by design
(binary-blob-bloat risk in git, per the plan's confirmed decision).
"""
import shutil
import subprocess
from pathlib import Path

import torch

STATE_ROOT = Path(__file__).parent / "user_states"


def _git(*args, cwd=STATE_ROOT):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def _ensure_repo():
    if not (STATE_ROOT / ".git").exists():
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        _git("init", "-q")
        _git("config", "user.email", "illation@local")
        _git("config", "user.name", "illation-user-state")


def user_dir(user_id):
    return STATE_ROOT / user_id


def init_user(user_id, model: "InterleavedStack"):
    """Creates a fresh per-user state directory from a model's current Internal +
    interleaving weights (e.g. right after Phase 2 training produces a shared starting
    point every new user begins from)."""
    _ensure_repo()
    d = user_dir(user_id)
    d.mkdir(parents=True, exist_ok=True)
    _save_weights(d, model)
    _git("add", str(d.relative_to(STATE_ROOT)))
    _git("commit", "-q", "-m", f"{user_id}: initial state", "--allow-empty")


def _save_weights(d: Path, model):
    torch.save({
        "internal": model.internal.state_dict(),
        "ext_reads_int": model.ext_reads_int.state_dict(),
        "int_reads_ext": model.int_reads_ext.state_dict(),
        "thought_start": model.thought_start,
    }, d / "internal_weights.pt")


def load_weights(user_id, model, device=None):
    """Loads a user's Internal + interleaving weights into model in-place (External stays
    whatever it already is -- shared/frozen, not touched here)."""
    d = user_dir(user_id)
    path = d / "internal_weights.pt"
    if not path.exists():
        raise FileNotFoundError(f"no saved state for user {user_id!r} at {path}")
    ckpt = torch.load(path, map_location=device)
    model.internal.load_state_dict(ckpt["internal"])
    model.ext_reads_int.load_state_dict(ckpt["ext_reads_int"])
    model.int_reads_ext.load_state_dict(ckpt["int_reads_ext"])
    with torch.no_grad():
        model.thought_start.copy_(ckpt["thought_start"])
    return model


def save_checkpoint(user_id, model, label=None):
    """Coarse, git-backed checkpoint: commit the user's CURRENT weights as a new point in
    their history. Call this at a checkpoint interval (every N sessions / explicit save),
    not after every turn."""
    _ensure_repo()
    d = user_dir(user_id)
    d.mkdir(parents=True, exist_ok=True)
    _save_weights(d, model)
    _git("add", str(d.relative_to(STATE_ROOT)))
    msg = f"{user_id}: checkpoint" + (f" ({label})" if label else "")
    _git("commit", "-q", "-m", msg, "--allow-empty")
    return _git("rev-parse", "HEAD").strip()


def history(user_id):
    """List this user's checkpoint history: [(commit_hash, message), ...], newest first."""
    _ensure_repo()
    rel = str(user_dir(user_id).relative_to(STATE_ROOT))
    out = _git("log", "--pretty=format:%H\t%s", "--", rel)
    if not out:
        return []
    return [tuple(line.split("\t", 1)) for line in out.splitlines()]


def revert(user_id, commit_ish):
    """Restore this user's weight file to the state at a given commit (or 'HEAD~1', etc.)."""
    _ensure_repo()
    rel = str(user_dir(user_id).relative_to(STATE_ROOT))
    _git("checkout", commit_ish, "--", rel)
    _git("commit", "-q", "-m", f"{user_id}: revert to {commit_ish}", "--allow-empty")


def import_weights(target_user_id, source_user_id=None, source_path=None):
    """'Plug and play' weight sharing: copy another user's (or an arbitrary file's)
    Internal+interleaving weights into target_user_id, committed as a normal checkpoint --
    NOT a destructive overwrite. target_user_id's own prior states remain in their git
    history; they can revert() back to their own weights at any time afterward."""
    _ensure_repo()
    assert (source_user_id is None) != (source_path is None), "give exactly one of source_user_id or source_path"
    src = user_dir(source_user_id) / "internal_weights.pt" if source_user_id else Path(source_path)
    if not src.exists():
        raise FileNotFoundError(src)
    d = user_dir(target_user_id)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, d / "internal_weights.pt")
    _git("add", str(d.relative_to(STATE_ROOT)))
    label = source_user_id or str(source_path)
    _git("commit", "-q", "-m", f"{target_user_id}: imported from {label}", "--allow-empty")
    return _git("rev-parse", "HEAD").strip()
