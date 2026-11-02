# Account-switch checkpoint — 2026-11-02

Code and scientific continuation guide: [research/evidence-aware-triage](https://github.com/appleweiping/applecider-agents/tree/research/evidence-aware-triage), [docs/continuation.md](https://github.com/appleweiping/applecider-agents/blob/research/evidence-aware-triage/docs/continuation.md).

This branch stores backup artifacts separately from reviewable source. Confirmation replay is incomplete: all 4,096 captures are retained, with 129,037 complete rows in the normally stopped retry. Earlier overlapping attempts are preserved and must not be counted as extra samples. No final confirmation aggregate exists.

## Public scientific checkpoint

`applecider-scientific-checkpoint-20261102.tar.gz` contains portable scientific artifacts, an inventory, checksums and explicit incomplete status. SHA256: `4533963497e0547995e9a7567269f0c33088a5be90da4d478e67866a060e0bd6`.

## Private continuation state

The three `.aesgcm.part*` files are an authenticated encrypted archive, split to respect GitHub file limits. Internal workflow configuration and research logs remain encrypted. The key is supplied only in the owner’s private chat; it is not in this branch. Keep it private.

Restore on a trusted machine:

```bash
sha256sum -c SHA256SUMS
cat private-research-checkpoint-20261102.aesgcm.part01 private-research-checkpoint-20261102.aesgcm.part02 private-research-checkpoint-20261102.aesgcm.part03 > private-research-checkpoint-20261102.aesgcm
sha256sum private-research-checkpoint-20261102.aesgcm
python -m pip install cryptography
read -rsp 'Private checkpoint key: ' CHECKPOINT_KEY; echo
export CHECKPOINT_KEY
python restore_checkpoint.py private-research-checkpoint-20261102.aesgcm private-research-checkpoint.tar.gz
unset CHECKPOINT_KEY
mkdir -p restored
tar -xzf private-research-checkpoint.tar.gz -C restored
```

The concatenated ciphertext SHA256 must be `ccef02946e7bf5d9be37ed2bf37ae68e7d01ce04109b70c1eefd1335254c4800`.
The decrypted archive SHA256 is `b00849ea985b86c326a848a5b687966cbcef1d1b23fc598f67fbeed63c2b9175`.

Read `restored/.research/applecider-agents/ACCOUNT_SWITCH_README.md` first, followed by `HANDOFF_ACCOUNT_SWITCH.md`. The archive preserves the private workflow version, reviews, frozen protocols and pause receipts. External public dependencies and the Python virtual environment are excluded; their pinned reconstruction instructions are retained. Clone the source branch separately. Do not automatically restart stopped workers or overwrite partial attempts.
