"""Prepare a zoo update; upload only with --publish."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from tigris.emitters.binary.reader import read_binary_plan
from tigris.zoo import REPOSITORY, artifact_path, digest, manifest_fields, validate_artifact, validate_catalog, version

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_build(directory):
    metadata = json.loads((directory / "artifact.json").read_text())
    if "published_at" in metadata or "manifest_sha256" in metadata or "withdrawn" in metadata:
        raise ValueError("build metadata must not contain publication fields")
    probe = dict(metadata, published_at="2000-01-01T00:00:00Z")
    validate_artifact(probe)
    paths = {item["path"] for item in metadata["files"]}
    if not {"readme.md", "license.txt", "evaluation.json", "example-input.bin", "example-output.bin"} <= paths:
        raise ValueError("build must include documentation, license, evaluation, and example I/O")
    for item in metadata["files"]:
        path = directory / item["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or nonregular file: {item['path']}")
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError(f"build checksum mismatch: {item['path']}")
    plan = read_binary_plan((directory / "model.tgrs").read_bytes())
    if plan["version"] != metadata["schema"] or plan["budget"] != metadata["memory"]["fast_bytes"]:
        raise ValueError("plan schema or budget disagrees with metadata")
    if (directory / "model.tgrs").stat().st_size != metadata["memory"]["flash_bytes"]:
        raise ValueError("plan size disagrees with metadata")
    evaluation = json.loads((directory / "evaluation.json").read_text())
    if evaluation != metadata["evaluation"]:
        raise ValueError("evaluation file disagrees with metadata")
    if evaluation.get("parity_passed") is not True or evaluation.get("test_windows", 0) <= 0:
        raise ValueError("evaluation must include successful runtime parity and held-out task measurements")
    tested = evaluation.get("tested_runtime_versions", [])
    if not isinstance(tested, list) or not tested:
        raise ValueError("build evaluation must record tested runtimes")
    for tested_version in tested:
        version(tested_version)
    if not isinstance(metadata.get("compatibility_note"), str) or not metadata["compatibility_note"].strip():
        raise ValueError("build must explain its runtime constraints")
    return metadata


def prepare(builds, catalog, output, *, now=None, withdraw=None, reason=None, update=None, metadata_update=None,
            card=False):
    validate_catalog(catalog)
    if output.exists():
        raise ValueError("staging destination already exists")
    updated = copy.deepcopy(catalog)
    existing = {item["id"] for item in updated["artifacts"]}
    pending = []
    for directory in builds:
        metadata = validate_build(directory)
        if metadata["id"] in existing:
            raise ValueError(f"artifact ID already exists: {metadata['id']}")
        existing.add(metadata["id"])
        pending.append((directory, metadata))
    if withdraw:
        matches = [item for item in updated["artifacts"] if item["id"] == withdraw]
        if not matches or not reason or not reason.strip():
            raise ValueError("withdrawal requires an existing artifact ID and a reason")
        matches[0]["withdrawn"] = reason
    if update:
        matches = [item for item in updated["artifacts"] if item["id"] == update]
        allowed = {"runtime", "tested_runtime_versions", "compatibility_note"}
        if not matches or not isinstance(metadata_update, dict) or not metadata_update or set(metadata_update) - allowed:
            raise ValueError("metadata update requires an existing ID and only compatibility or validation fields")
        if "runtime" in metadata_update:
            note = metadata_update.get("compatibility_note")
            if not isinstance(note, str) or not note.strip():
                raise ValueError("runtime constraint updates require a compatibility_note")
        tested = metadata_update.get("tested_runtime_versions", matches[0]["tested_runtime_versions"])
        if not isinstance(tested, list) or not all(v in tested for v in matches[0]["tested_runtime_versions"]):
            raise ValueError("metadata updates must preserve recorded runtime validations")
        matches[0].update(copy.deepcopy(metadata_update))
    validate_catalog(updated)
    published = now or datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    output.mkdir(parents=True)
    for directory, metadata in pending:
        metadata["published_at"] = published
        target = output / artifact_path(metadata)
        target.mkdir(parents=True)
        for item in metadata["files"]:
            dest = target / item["path"]
            shutil.copyfile(directory / item["path"], dest)
            if digest(dest) != item["sha256"] or dest.stat().st_size != item["size"]:
                raise ValueError("build changed while staging")
        manifest = target / "manifest.json"
        write_json(manifest, metadata)
        updated["artifacts"].append(dict(metadata, manifest_sha256=digest(manifest), withdrawn=None,
                                         tested_runtime_versions=metadata["evaluation"]["tested_runtime_versions"]))
    validate_catalog(updated)
    write_json(output / "catalog.json", updated)
    if card or not catalog["artifacts"]:
        for name in ("README.md", "LICENSE"):
            shutil.copyfile(ROOT / "hub" / name, output / name)
    return updated


def remote_catalog(repository):
    info = HfApi(token=False).repo_info(repository, revision="main")
    files = {item.rfilename for item in info.siblings}
    if "catalog.json" in files:
        path = hf_hub_download(repository, "catalog.json", revision=info.sha, token=False)
        catalog = json.loads(Path(path).read_text())
    else:
        catalog = {"format_version": 1, "artifacts": []}
    return info.sha, files, catalog


def publish(stage, repository, parent, existing_files, original_catalog, *, card=False):
    # Optimistic concurrency protects the catalog across all publishing clients.
    staged = json.loads((stage / "catalog.json").read_text())
    validate_catalog(staged)
    old_ids = {item["id"] for item in original_catalog["artifacts"]}
    by_id = {item["id"]: item for item in staged["artifacts"]}
    for old in original_catalog["artifacts"]:
        current = by_id.get(old["id"])
        if current is None or manifest_fields(current) != manifest_fields(old) or current["manifest_sha256"] != old["manifest_sha256"]:
            raise ValueError("published artifact contents, identity, and date must remain unchanged")
        if not set(old["tested_runtime_versions"]) <= set(current["tested_runtime_versions"]):
            raise ValueError("published runtime validations must remain recorded")
    allowed = {"catalog.json"}
    if card or not original_catalog["artifacts"]:
        allowed.update(("README.md", "LICENSE"))
    for item in staged["artifacts"]:
        if item["id"] not in old_ids:
            prefix = artifact_path(item) + "/"
            if any(path.startswith(prefix) for path in existing_files):
                raise ValueError(f"artifact directory already exists: {item['id']}")
            manifest = stage / prefix / "manifest.json"
            if digest(manifest) != item["manifest_sha256"] or manifest_fields(json.loads(manifest.read_text())) != manifest_fields(item):
                raise ValueError("staged manifest changed before publication")
            allowed.add(prefix + "manifest.json")
            for entry in item["files"]:
                path = stage / prefix / entry["path"]
                if path.is_symlink() or digest(path) != entry["sha256"] or path.stat().st_size != entry["size"]:
                    raise ValueError("staged artifact changed before publication")
                allowed.add(prefix + entry["path"])
    actual = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
    if actual != allowed:
        raise ValueError("staging directory contains missing or unexpected files")
    operations = [CommitOperationAdd(path_in_repo=path.relative_to(stage).as_posix(), path_or_fileobj=path)
                  for path in sorted(stage.rglob("*")) if path.is_file()]
    message = ("docs: update the model card" if card and staged == original_catalog
               else "feat: update compiled model catalog")
    return HfApi().create_commit(
        repo_id=repository, revision="main", parent_commit=parent, operations=operations,
        commit_message=message, commit_description="",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("builds", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, help="Local catalog for an offline preparation")
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--withdraw")
    parser.add_argument("--reason")
    parser.add_argument("--update", help="Artifact ID whose catalog metadata should change")
    parser.add_argument("--metadata", type=Path, help="JSON containing runtime constraints and/or tested_runtime_versions")
    parser.add_argument("--card", action="store_true", help="Also upload hub/README.md and hub/LICENSE")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    try:
        if bool(args.withdraw) != bool(args.reason) or bool(args.update) != bool(args.metadata):
            raise ValueError("pair --withdraw with --reason, and --update with --metadata")
        if not args.builds and not args.withdraw and not args.update and not args.card:
            raise ValueError("provide builds, a withdrawal, a metadata update, or --card")
        if args.publish and args.catalog:
            raise ValueError("--publish must read the current remote catalog")
        if args.catalog:
            original = json.loads(args.catalog.read_text())
        else:
            parent, files, original = remote_catalog(args.repository)
        if args.publish:
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
            if status.strip():
                raise ValueError("commit the publishing source before publishing")
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            for directory in args.builds:
                source = validate_build(directory)["source"]
                paths = dict(source.get("recipe_files", {}))
                paths[source["recipe"]] = source["recipe_sha256"]
                if source.get("recipe_commit") != commit:
                    raise ValueError("rebuild artifacts from the checked-out recipe commit before publishing")
                for name, expected in paths.items():
                    path = (ROOT / name).resolve()
                    if not path.is_relative_to(ROOT) or digest(path) != expected:
                        raise ValueError("recipe source changed since the artifact was built")
        updated = prepare(args.builds, original, args.output, withdraw=args.withdraw, reason=args.reason,
                          update=args.update, metadata_update=json.loads(args.metadata.read_text()) if args.metadata else None,
                          card=args.card)
        print(f"Prepared {len(updated['artifacts'])} catalog entries in {args.output}")
        if args.publish:
            result = publish(args.output, args.repository, parent, files, original, card=args.card)
            print(result.commit_url)
        else:
            print("Preparation only; nothing uploaded.")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
