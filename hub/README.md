---
license: other
license_name: per-model-licenses
license_link: LICENSE
tags:
- tigris
- embedded
---

# TiGrIS model zoo

Precompiled `.tgrs` models with runtime requirements, input/output conventions,
evaluation results, and example tensors. Each model's artifacts are under
`models/<model>/artifacts/<artifact-id>/`.

Each artifact's `readme.md` documents the model's task, input preprocessing,
output units, evaluation results and limitations. `example-output.bin` is the
reference output the recipe compared the runtime against for `example-input.bin`;
`evaluation.json` records how closely the runtime reproduced the reference.

```bash
python -m venv .venv
source .venv/bin/activate
pip install tigris-ml
tigris zoo list
tigris zoo fetch electricity-hourly -o downloaded-model
tigris codegen downloaded-model/model.tgrs --format core -o model.c
```

Downloads need no account. Filters select compatible builds before choosing the
newest publication. Supply the runtime separately within the current range recorded
in `download.json`. A null maximum means no known upper cutoff; tested releases
are listed separately. The original build manifest remains unchanged when catalog
compatibility records are updated. Fast/slow arena requirements exclude external buffers, stack,
and runtime metadata. Model quality and evaluation scope are documented per build.

Model licenses differ; inspect the downloaded `license.txt` and source provenance.
