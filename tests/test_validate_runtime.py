import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tigris.zoo import Zoo

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = load("publish", "scripts/publish.py")
validator = load("validate_runtime", "scripts/validate_runtime.py")


@pytest.fixture
def published(build, tmp_path):
    stage = tmp_path / "stage"
    publisher.prepare([build], {"format_version": 1, "artifacts": []}, stage)
    return Zoo(catalog=stage / "catalog.json")


@pytest.fixture
def runtime(tmp_path):
    path = tmp_path / "runtime"
    path.mkdir()
    for args in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "release"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=path, check=True)
    return path


def gates(result=None, error=None):
    def revalidate(plan, manifest, runtime, output, dataset):
        assert plan.name == "model.tgrs" and manifest["id"]
        if error:
            raise error
        return result or {"parity_passed": True}
    return SimpleNamespace(revalidate=revalidate)


def test_a_pass_stages_the_release_after_the_recorded_history(published, runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "recipe_module", lambda manifest: gates({"parity_passed": True}))
    staged = validator.validate(published, published.artifacts, "0.10.0", tmp_path / "out", runtime, tmp_path)
    artifact = published.artifacts[0]
    assert staged == [artifact["id"]]
    update = json.loads((tmp_path / "out" / artifact["id"] / "metadata.json").read_text())
    assert update == {"tested_runtime_versions": artifact["tested_runtime_versions"] + ["0.10.0"]}
    record = json.loads((tmp_path / "out" / artifact["id"] / "validation.json").read_text())
    assert record["runtime_version"] == "0.10.0" and len(record["runtime_commit"]) == 40


def test_a_failed_gate_stages_nothing(published, runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "recipe_module", lambda manifest: gates(error=ValueError("parity")))
    with pytest.raises(ValueError, match="parity"):
        validator.validate(published, published.artifacts, "0.10.0", tmp_path / "out", runtime, tmp_path)
    assert not list((tmp_path / "out").rglob("metadata.json"))


def test_a_recorded_release_is_not_validated_again(published, runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "recipe_module", lambda manifest: pytest.fail("must not rerun"))
    recorded = published.artifacts[0]["tested_runtime_versions"][0]
    assert validator.validate(published, published.artifacts, recorded, tmp_path / "out", runtime, tmp_path) == []


def test_a_release_outside_the_declared_range_is_refused(published, runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "recipe_module", lambda manifest: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="outside its declared range"):
        validator.validate(published, published.artifacts, "0.0.1", tmp_path / "out", runtime, tmp_path)


def test_only_repository_recipes_can_revalidate():
    with pytest.raises(ValueError, match="no recipe"):
        validator.recipe_module({"source": {"recipe": "../outside.py"}})
