# TiGrIS model zoo

Build recipes and publication tools for trained `.tgrs` models. Published plans
and their per-model licenses are available from
[the model repository](https://huggingface.co/raws-labs/tigris-zoo).

Use `tigris zoo list` to inspect published models and `tigris zoo fetch MODEL` to
download a plan with its runtime requirements. Downloads require no account.

For a local build, use Python 3.12 and install the compiler and build dependencies
in a project environment:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -c requirements-build.txt \
  'tigris-ml[dev] @ git+https://github.com/raws-labs/tigris.git@cafc9315e1386dd39639370a4b8609e5413f5108'
mkdir -p .build
(cd .build && ../.venv/bin/python ../recipes/electricity.py --output electricity)
.venv/bin/python scripts/publish.py .build/electricity/*k \
  --catalog catalog.json --output .build/staged
.venv/bin/tigris zoo --catalog .build/staged/catalog.json list
```

The recipe downloads the dataset and pinned compiler/runtime sources on its first
run. Builds need a C compiler and CMake. Host training and evaluation should run
under an appropriate resource limit; thread counts are capped by the workflow.
Artifacts include task measurements, example tensors, and input/output conventions.

Publication and validation are described in [CONTRIBUTING.md](CONTRIBUTING.md).
