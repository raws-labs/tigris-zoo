import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
from click.testing import CliRunner

from tigris import SCHEMA_VERSION
from tigris.cli import cli
from tigris.zoo import Zoo, artifact_path, digest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish", ROOT / "scripts/publish.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


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


def test_prepare_fetch_codegen_and_reject_duplicate_id(build, tmp_path):
    catalog = {"format_version": 1, "artifacts": []}
    stage = tmp_path / "stage"
    updated = publisher.prepare([build], catalog, stage, now="2026-01-01T00:00:00Z")
    assert not catalog["artifacts"]
    source = Zoo(catalog=stage / "catalog.json")
    plan = source.fetch(source.artifacts[0], tmp_path / "download")
    result = CliRunner().invoke(cli, ["codegen", str(plan), "--format", "core", "-o", str(tmp_path / "model.c")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "model.h").is_file()
    with pytest.raises(ValueError, match="already exists"):
        publisher.prepare([build], updated, tmp_path / "second")


@pytest.mark.parametrize("damage", ["checksum", "schema", "evaluation", "runtime", "maximum", "timestamp"])
def test_reject_invalid_build(build, tmp_path, damage):
    metadata_path = build / "artifact.json"
    metadata = json.loads(metadata_path.read_text())
    if damage == "checksum":
        (build / "model.tgrs").write_bytes(b"bad")
    elif damage == "schema":
        metadata["schema"] += 1
    elif damage == "evaluation":
        metadata["evaluation"]["parity_passed"] = False
    elif damage == "runtime":
        metadata["runtime"]["min"] = "invalid"
    elif damage == "maximum":
        metadata["runtime"]["max"] = "0.9.0"
    else:
        metadata["published_at"] = "2026-01-01T00:00:00Z"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        publisher.prepare([build], {"format_version": 1, "artifacts": []}, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_withdrawal_preserves_immutable_metadata(build, tmp_path):
    initial = publisher.prepare([build], {"format_version": 1, "artifacts": []}, tmp_path / "initial")
    updated = publisher.prepare([], initial, tmp_path / "withdrawal", withdraw="example-a", reason="bad output")
    old = copy.deepcopy(initial["artifacts"][0])
    old["withdrawn"] = "bad output"
    assert updated["artifacts"] == [old]
    assert list((tmp_path / "withdrawal").iterdir()) == [tmp_path / "withdrawal/catalog.json"]


def test_metadata_update_changes_only_catalog_and_preserves_artifact(build, tmp_path, monkeypatch):
    initial_stage = tmp_path / "initial"
    initial = publisher.prepare([build], {"format_version": 1, "artifacts": []}, initial_stage)
    item = initial["artifacts"][0]
    original_manifest = (initial_stage / artifact_path(item) / "manifest.json").read_bytes()
    stage = tmp_path / "update"
    updated = publisher.prepare([], initial, stage, update=item["id"], metadata_update={
        "tested_runtime_versions": ["0.9.1", "0.10.0"],
    })
    changed = updated["artifacts"][0]
    assert changed["runtime"] == {"min": "0.9.1", "max": None}
    assert changed["published_at"] == item["published_at"]
    assert changed["manifest_sha256"] == item["manifest_sha256"]
    assert changed["files"] == item["files"]
    assert list(stage.iterdir()) == [stage / "catalog.json"]
    calls = []
    monkeypatch.setattr(publisher, "HfApi", lambda: SimpleNamespace(create_commit=lambda **kw: calls.append(kw)))
    publisher.publish(stage, "example/zoo", "a" * 40, set(), initial)
    assert [op.path_in_repo for op in calls[0]["operations"]] == ["catalog.json"]
    publisher.write_json(initial_stage / "catalog.json", updated)
    source = Zoo(catalog=initial_stage / "catalog.json")
    source.fetch(source.artifacts[0], tmp_path / "download")
    assert (tmp_path / "download/manifest.json").read_bytes() == original_manifest
    receipt = json.loads((tmp_path / "download/download.json").read_text())
    assert receipt["tested_runtime_versions"] == ["0.9.1", "0.10.0"]


def test_known_cutoff_can_be_recorded_without_rebuilding(build, tmp_path):
    from tigris.zoo import select

    initial = publisher.prepare([build], {"format_version": 1, "artifacts": []}, tmp_path / "initial")
    updated = publisher.prepare([], initial, tmp_path / "update", update="example-a", metadata_update={
        "runtime": {"min": "0.9.1", "max": "0.9.3"},
        "compatibility_note": "Fixture: later runtime drops the required schema.",
    })
    assert select(updated["artifacts"], runtime="0.9.3")
    assert not select(updated["artifacts"], runtime="0.10.0")
    assert updated["artifacts"][0]["tested_runtime_versions"] == ["0.9.1"]


@pytest.mark.parametrize("update", [
    {"published_at": "2027-01-01T00:00:00Z"},
    {"manifest_sha256": "b" * 64},
    {"files": []},
    {"runtime": {"min": "0.9.1", "max": "0.9.3"}},
    {"tested_runtime_versions": []},
    {"tested_runtime_versions": ["0.9.1", "invalid"]},
])
def test_invalid_metadata_update_is_rejected(build, tmp_path, update):
    initial = publisher.prepare([build], {"format_version": 1, "artifacts": []}, tmp_path / "initial")
    with pytest.raises(ValueError):
        publisher.prepare([], initial, tmp_path / "update", update="example-a", metadata_update=update)
    assert not (tmp_path / "update").exists()


def test_publish_rejects_changed_existing_artifact(build, tmp_path, monkeypatch):
    initial = publisher.prepare([build], {"format_version": 1, "artifacts": []}, tmp_path / "initial")
    stage = tmp_path / "update"
    updated = publisher.prepare([], initial, stage, update="example-a", metadata_update={
        "tested_runtime_versions": ["0.9.1", "0.10.0"],
    })
    updated["artifacts"][0]["published_at"] = "2027-01-01T00:00:00Z"
    publisher.write_json(stage / "catalog.json", updated)
    monkeypatch.setattr(publisher, "HfApi", lambda: pytest.fail("must not publish altered artifacts"))
    with pytest.raises(ValueError, match="must remain unchanged"):
        publisher.publish(stage, "example/zoo", "a" * 40, set(), initial)


def test_publish_is_one_guarded_commit_without_deletions(build, tmp_path, monkeypatch):
    original = {"format_version": 1, "artifacts": []}
    stage = tmp_path / "stage"
    publisher.prepare([build], original, stage)
    calls = []

    def commit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(commit_url="https://example.test/commit")

    monkeypatch.setattr(publisher, "HfApi", lambda: SimpleNamespace(create_commit=commit))
    publisher.publish(stage, "example/zoo", "a" * 40, {"README.md", "LICENSE"}, original)
    assert len(calls) == 1
    assert calls[0]["parent_commit"] == "a" * 40
    assert calls[0]["revision"] == "main"
    assert {type(op).__name__ for op in calls[0]["operations"]} == {"CommitOperationAdd"}
    assert "catalog.json" in {op.path_in_repo for op in calls[0]["operations"]}


def test_publish_refuses_orphan_directory_and_changed_stage(build, tmp_path, monkeypatch):
    original = {"format_version": 1, "artifacts": []}
    stage = tmp_path / "stage"
    updated = publisher.prepare([build], original, stage)
    prefix = artifact_path(updated["artifacts"][0])
    monkeypatch.setattr(publisher, "HfApi", lambda: pytest.fail("must not write to HF"))
    with pytest.raises(ValueError, match="already exists"):
        publisher.publish(stage, "example/zoo", "a" * 40, {prefix + "/old.bin"}, original)
    (stage / prefix / "model.tgrs").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed before publication"):
        publisher.publish(stage, "example/zoo", "a" * 40, set(), original)


def test_conflicting_remote_commit_is_not_retried(build, tmp_path, monkeypatch):
    original = {"format_version": 1, "artifacts": []}
    stage = tmp_path / "stage"
    publisher.prepare([build], original, stage)
    calls = []

    def commit(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("parent commit changed")

    monkeypatch.setattr(publisher, "HfApi", lambda: SimpleNamespace(create_commit=commit))
    with pytest.raises(RuntimeError, match="parent commit changed"):
        publisher.publish(stage, "example/zoo", "a" * 40, set(), original)
    assert len(calls) == 1


def test_command_defaults_to_preparation_only(build, tmp_path, monkeypatch):
    original = {"format_version": 1, "artifacts": []}
    stage = tmp_path / "stage"
    monkeypatch.setattr(publisher, "remote_catalog", lambda repo: ("a" * 40, set(), original))
    monkeypatch.setattr(publisher, "publish", lambda *args: pytest.fail("must not publish without --publish"))
    monkeypatch.setattr(sys, "argv", ["publish.py", str(build), "--output", str(stage)])
    publisher.main()
    assert len(json.loads((stage / "catalog.json").read_text())["artifacts"]) == 1


def test_publish_requires_committed_source(build, tmp_path, monkeypatch):
    original = {"format_version": 1, "artifacts": []}
    monkeypatch.setattr(publisher, "remote_catalog", lambda repo: ("a" * 40, set(), original))
    monkeypatch.setattr(publisher, "publish", lambda *args: pytest.fail("must not publish uncommitted source"))
    monkeypatch.setattr(publisher.subprocess, "check_output", lambda *args, **kwargs: " M recipes/electricity.py\n")
    monkeypatch.setattr(sys, "argv", ["publish.py", str(build), "--output", str(tmp_path / "stage"), "--publish"])
    with pytest.raises(SystemExit) as exc:
        publisher.main()
    assert exc.value.code == 1
