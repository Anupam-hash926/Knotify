# Local run outputs

`storage.save_report()` writes `analyses/<uuid>.json` without overwriting an existing report.
`storage.save_review()` writes `reviews/<uuid>.json` only after validating the saved analysis.
The current UI previews/downloads JSON; it does not yet expose save/review controls.

Future evaluation directories can contain predictions, metrics and configuration manifests.
Everything except this README is ignored by Git; no measured results exist yet.
