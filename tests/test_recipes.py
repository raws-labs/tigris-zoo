import importlib.util
from pathlib import Path

import pytest

from tigris.zoo import digest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("int8_common", ROOT / "recipes" / "int8_common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def test_fetch_tries_mirrors_in_turn(tmp_path):
    good = tmp_path / "good.bin"
    good.write_bytes(b"pinned")
    other = tmp_path / "other.bin"
    other.write_bytes(b"changed")
    missing = (tmp_path / "missing.bin").as_uri()
    path = common.fetch(missing, tmp_path / "data" / "file.bin", digest(good),
                        (other.as_uri(), good.as_uri()))
    assert path.read_bytes() == b"pinned"
    assert not path.with_suffix(".part").exists()


def test_fetch_names_every_failed_source(tmp_path):
    other = tmp_path / "other.bin"
    other.write_bytes(b"changed")
    missing = (tmp_path / "missing.bin").as_uri()
    with pytest.raises(ValueError, match="missing.bin.*\n.*other.bin: download checksum mismatch"):
        common.fetch(missing, tmp_path / "data" / "file.bin", "0" * 64, (other.as_uri(),))
    assert not (tmp_path / "data" / "file.part").exists()
