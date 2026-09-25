import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
from click.testing import CliRunner

from tigris import SCHEMA_VERSION
from tigris.cli import cli
from tigris.zoo import digest


@pytest.fixture
def build(tmp_path):
    directory = tmp_path / "build"
    directory.mkdir()
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Gemm", ["input", "weight"], ["output"], transB=1)], "example",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(np.array([[1, 2]], dtype=np.float32), "weight")],
    ), opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    source = tmp_path / "example.onnx"
    onnx.save(model, source)
    result = CliRunner().invoke(cli, ["compile", str(source), "--xip", "-m", "1K", "-o", str(directory / "model.tgrs")])
    assert result.exit_code == 0, result.output
    evaluation = {"parity_passed": True, "test_windows": 1, "tested_runtime_versions": ["0.9.1"]}
    for name in ("readme.md", "license.txt", "example-input.bin", "example-output.bin"):
        (directory / name).write_bytes(b"example")
    (directory / "evaluation.json").write_text(json.dumps(evaluation))
    metadata = {
        "format_version": 1, "id": "example-a", "model": "example", "category": "regression",
        "quantization": "float32", "schema": SCHEMA_VERSION, "backends": ["reference"],
        "runtime": {"min": "0.9.1", "max": None},
        "compatibility_note": "Reference backend requirements; no known upper cutoff.",
        "memory": {"fast_bytes": 1024, "slow_bytes": 0, "flash_bytes": (directory / "model.tgrs").stat().st_size},
        "inputs": [{"name": "input", "shape": [1, 2], "dtype": "float32"}],
        "outputs": [{"name": "output", "shape": [1, 1], "dtype": "float32"}],
        "license": "apache-2.0", "compiler": {"commit": "a" * 40},
        "source": {"recipe_commit": "b" * 40}, "evaluation": evaluation,
        "files": [{"path": p.name, "size": p.stat().st_size, "sha256": digest(p)} for p in sorted(directory.iterdir())],
    }
    (directory / "artifact.json").write_text(json.dumps(metadata))
    return directory
