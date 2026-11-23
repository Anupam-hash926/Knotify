# Local datasets

Frozen GitHub snapshots (`raw/`), split manifests (`splits/`), duplicate evidence
(`duplicates/`) and annotation/review files (`annotations/`) are written here by the `knotify`
CLI; see IMPLEMENTATION_STATUS.md for the commands. All contents except this README are ignored by Git. Synthetic test cases
live in `tests/`; they are not training data or measured project results.

Suggested future layout: `raw/`, `processed/`, `annotations/`. Record collection date,
repository IDs, content hashes, label mappings, split boundaries and annotation provenance.
