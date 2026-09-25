# tigris-zoo

Build recipes and publishing tools for precompiled model artifacts.
Read CONTRIBUTING.md before building or changing published content.

## Branches
- `main` is the only branch and is protected: changes land through pull
  requests with the `Tests` check passing.

## Publishing
- Prepare locally by default. Publishing requires an explicit user request.
- Use scripts/publish.py or the manual publishing workflow.
- Publish from `main` only; the workflow's publish steps run in the
  `hf-publish` environment, which holds `HF_TOKEN` and deploys only from `main`.
- Change the model card in hub/README.md and publish it with `--card`.
- Never upload artifact files or edit the remote catalog directly.
- Publication requires a clean, committed checkout matching build provenance.
- Preserve published artifacts, manifests, IDs, and publication dates.
- Withdraw defective artifacts through the publisher; never delete them.
- Update compatibility and validation records through catalog metadata updates.
- Runtime maxima describe known cutoffs, never the newest tested release.
- Record additional tested runtimes only after validating existing plan bytes.
- Keep credentials out of tracked files and command output.

## Verification
- Use the project .venv and dependency constraints in requirements-build.txt.
- Run pytest tests and ruff check --select E4,E7,E9,F recipes scripts tests.
- Preserve dataset checksums, source pins, task-quality and runtime-parity gates.
- Keep generated artifacts in .build; publish binaries to HF.
