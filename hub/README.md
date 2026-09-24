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

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'tigris-ml @ git+https://github.com/raws-labs/tigris.git@cafc9315e1386dd39639370a4b8609e5413f5108'
tigris zoo list
tigris zoo fetch electricity-hourly -o downloaded-model
tigris codegen downloaded-model/model.tgrs --format core -o model.c
```

The electricity model predicts the next 24 hourly readings from the preceding
168 hours. It is evaluated on a chronological holdout from one household, with
daily and weekly persistence comparisons. See the downloaded `readme.md` for
normalization, units, evaluation results, and limitations.

Downloads need no account. Filters select compatible builds before choosing the
newest publication. Supply the runtime separately within the current range recorded
in `download.json`. A null maximum means no known upper cutoff; tested releases
are listed separately. The original build manifest remains unchanged when catalog
compatibility records are updated. Fast/slow arena requirements exclude external buffers, stack,
and runtime metadata. Model quality and evaluation scope are documented per build.

Model licenses differ; inspect the downloaded `license.txt` and source provenance.
