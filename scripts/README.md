# Offline scripts

Reserved for thin import/training/evaluation wrappers. Implementations belong in `src/knotify`,
not duplicated in scripts or Streamlit. Working commands live in the installed `knotify` CLI: `audit`, `import`, `split`, `train`,
`duplicates`, `label-sheet`, `doctor`, `schema`. `calibrate` and `evaluate` are still reserved
and exit with code 2.
