# Building and publishing

Branches start from `develop`; changes are reviewed before publication. Recipes
pin sources and verify dataset hashes. They evaluate trained models on held-out
data and compare each compiled variant with ONNX Runtime on that same data.
Hardware execution is a separate manual check, never part of CI.

`scripts/publish.py` validates build metadata and file hashes, checks the plan's
schema and budget, and stages a complete catalog update. Without `--publish` it
does not write to HF. `--catalog` selects an existing local catalog for offline
preparation. Omit it to prepare against the current HF catalog.

Only a deliberate invocation with `--publish` uploads. It requires a clean,
committed recipe checkout matching the build provenance and HF write access to
the model repository. Supply credentials through `HF_TOKEN` or a local HF login.
Never put credentials in recipes. The manually dispatched workflow defaults to
preparation only and serializes runs. Merges do not publish.

The model card is `hub/README.md` with `hub/LICENSE`. `--card` stages both, and
with `--publish` uploads them together with the unchanged catalog; the
workflow's `card` input runs only that step.

The publisher assigns timestamps once, rejects reused artifact IDs or existing
artifact directories, and uploads files and catalog in a single commit guarded
by the previous HF commit. If another publisher wins, prepare again against the
new catalog. Files already published are retained. To withdraw a defective build:

```bash
.venv/bin/python scripts/publish.py --withdraw ARTIFACT_ID \
  --reason 'Description of the defect' --output .build/withdrawal --publish
```

Withdrawal updates catalog status without touching artifact files or dates.
An explicit-ID download reports the withdrawal reason.

Catalog and manifest format versions are independent of plan schema versions.
Each catalog entry duplicates the immutable build fields and adds the manifest's
SHA-256, a nullable withdrawal reason, current runtime constraints, a compatibility
note, and tested runtime versions. Runtime bounds are inclusive; a null maximum
means no upper cutoff is known. Derive bounds from actual requirements and known
incompatibilities, never from the newest tested release.

Runtime constraints and validation history can change without rebuilding or
republishing artifact files. The original manifest remains intact. Downloads
preserve the current catalog records in `download.json`, which supplies the
effective dependency information; `manifest.json` is the build-time snapshot.

After validating existing plan bytes on another runtime release, update the
catalog's `tested_runtime_versions` list. Preserve earlier validation entries.
For an actual incompatibility, update `runtime` and explain it in
`compatibility_note`. Put only the changed fields in a JSON file, then prepare:

```bash
.venv/bin/python scripts/publish.py --update ARTIFACT_ID \
  --metadata compatibility.json --output .build/compatibility-update
```

Add `--publish` only to upload the catalog change. Neither operation changes the
artifact ID, publication date, plan, manifest, or file checksums. A range match
alone is not evidence of testing on every runtime in that range.

Provide multiple memory budgets when they produce meaningfully different
execution plans or measured latency. The electricity recipe emits only its
1 KiB fast-arena variant; a larger arena does not change its execution schedule.

Run `python -m pytest tests` and `ruff check recipes scripts tests` in the project
environment. Publication requires the recipe's task-quality and parity gates.
