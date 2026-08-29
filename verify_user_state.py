"""Verification (per the approved plan): user_state.py checkpoint/revert/import round-
trip. Creates two dummy per-user weight states, checkpoints both, imports one into the
other, confirms the target's git history still contains its own pre-import commit and
`git checkout` back to it restores the original weights byte-for-byte."""
import shutil

import torch

import user_state
from internal_model import InternalConfig
from external_model import ExternalConfig
from interleave import InterleavedStack


def weights_equal(model_a, model_b):
    sa, sb = model_a.internal.state_dict(), model_b.internal.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def main():
    if user_state.STATE_ROOT.exists():
        shutil.rmtree(user_state.STATE_ROOT)

    torch.manual_seed(0)
    internal_cfg = InternalConfig(vocab_size=20, n_layer=2, n_head=4, n_kv_head=4, n_embd=16, max_seq_len=16)
    external_cfg = ExternalConfig(vocab_size=30, n_layer=2, n_head=4, n_kv_head=4, n_embd=24, max_seq_len=16)

    model_alice = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4)
    torch.manual_seed(1)
    model_bob = InterleavedStack(internal_cfg, external_cfg, cross_n_head=4)
    assert not weights_equal(model_alice, model_bob), "sanity: alice and bob must start different"

    user_state.init_user("alice", model_alice)
    user_state.init_user("bob", model_bob)

    # Alice trains a bit (simulate online adaptation): perturb her weights, checkpoint.
    with torch.no_grad():
        for p in model_alice.internal.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    alice_pretrained_state = {k: v.clone() for k, v in model_alice.internal.state_dict().items()}
    commit_before_import = user_state.save_checkpoint("alice", model_alice, label="post-training")

    bob_original_state = {k: v.clone() for k, v in model_bob.internal.state_dict().items()}

    # Bob imports Alice's weights ("plug and play").
    user_state.import_weights("bob", source_user_id="alice")
    user_state.load_weights("bob", model_bob)
    bob_after_import = model_bob.internal.state_dict()
    imported_matches_alice = all(torch.equal(bob_after_import[k], alice_pretrained_state[k]) for k in bob_after_import)
    print(f"bob's weights after import match alice's: {imported_matches_alice}")

    # Bob's history still has his own pre-import commit -- revert back to it.
    bob_history = user_state.history("bob")
    print(f"bob's history ({len(bob_history)} commits):")
    for h, msg in bob_history:
        print(f"  {h[:8]} {msg}")
    assert len(bob_history) >= 2, "bob should have both his init commit and the import commit"

    pre_import_commit = bob_history[-1][0]  # oldest = his own init state
    user_state.revert("bob", pre_import_commit)
    user_state.load_weights("bob", model_bob)
    bob_reverted_state = model_bob.internal.state_dict()
    reverted_matches_original = all(torch.equal(bob_reverted_state[k], bob_original_state[k]) for k in bob_reverted_state)
    print(f"bob's weights after revert match his original: {reverted_matches_original}")

    ok = imported_matches_alice and reverted_matches_original
    print()
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
