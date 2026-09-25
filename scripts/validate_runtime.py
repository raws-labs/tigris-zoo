"""Validate published artifacts on a runtime release; stage catalog metadata updates."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

from tigris.zoo import REPOSITORY, Zoo, select, version

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPOSITORY = "https://github.com/raws-labs/tigris-runtime.git"


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def runtime_checkout(release, path):
    """A clean checkout of the runtime release tag, verified against the tag."""
    tag = f"v{release}"
    if not path.exists():
        git("clone", "--quiet", "--no-checkout", RUNTIME_REPOSITORY, str(path))
        git("checkout", "--quiet", "--detach", tag, cwd=path)
    if git("rev-parse", "HEAD", cwd=path) != git("rev-parse", f"{tag}^{{commit}}", cwd=path):
        raise ValueError(f"runtime checkout is not at {tag}")
    if git("status", "--porcelain", cwd=path):
        raise ValueError("runtime checkout must be clean")
    return path.resolve()


def recipe_module(manifest):
    path = (ROOT / manifest["source"]["recipe"]).resolve()
    if not path.is_relative_to(ROOT / "recipes") or not path.is_file():
        raise ValueError(f"artifact names no recipe in this repository: {manifest['source']['recipe']}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "revalidate"):
        raise ValueError(f"{manifest['source']['recipe']} cannot revalidate published plans")
    return module


def validate(zoo, artifacts, release, output, runtime, dataset):
    """Run each artifact's recipe gates on `release`; stage one metadata update per pass."""
    staged = []
    for artifact in artifacts:
        if release in artifact["tested_runtime_versions"]:
            print(f"{artifact['id']}: runtime {release} already recorded")
            continue
        if not select([artifact], runtime=release):
            raise ValueError(f"{artifact['id']}: runtime {release} is outside its declared range")
        workdir = output / artifact["id"]
        download = workdir / "download"
        zoo.fetch(artifact, download)
        manifest = json.loads((download / "manifest.json").read_text())
        evaluation = recipe_module(manifest).revalidate(
            download / "model.tgrs", manifest, runtime, workdir / "work", dataset)
        (workdir / "validation.json").write_text(json.dumps(
            dict(evaluation, runtime_version=release, runtime_commit=git("rev-parse", "HEAD", cwd=runtime)),
            indent=2, sort_keys=True) + "\n")
        update = {"tested_runtime_versions": artifact["tested_runtime_versions"] + [release]}
        (workdir / "metadata.json").write_text(json.dumps(update, indent=2) + "\n")
        staged.append(artifact["id"])
        print(f"{artifact['id']}: passed on runtime {release}; staged {workdir / 'metadata.json'}")
    return staged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, help="Runtime release to validate, such as 0.10.0")
    parser.add_argument("--artifact", action="append", help="Artifact ID; default: every active artifact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, help="Local catalog snapshot instead of the model repository")
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--runtime-source", type=Path, help="Existing checkout of the runtime release tag")
    parser.add_argument("--dataset", type=Path, default=ROOT / ".build/electricity.zip")
    args = parser.parse_args()
    try:
        version(args.runtime)
        if args.output.exists():
            raise ValueError("output already exists; use a new directory")
        args.output.mkdir(parents=True)
        zoo = Zoo(catalog=args.catalog, repository=args.repository)
        active = [item for item in zoo.artifacts if item["withdrawn"] is None]
        if args.artifact:
            unknown = set(args.artifact) - {item["id"] for item in active}
            if unknown:
                raise ValueError(f"unknown or withdrawn artifact: {', '.join(sorted(unknown))}")
            active = [item for item in active if item["id"] in args.artifact]
        runtime = runtime_checkout(args.runtime, args.runtime_source or args.output / "runtime")
        staged = validate(zoo, active, args.runtime, args.output, runtime, args.dataset.resolve())
        print(f"Staged {len(staged)} metadata update(s) in {args.output}")
    except (ValueError, OSError, AssertionError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
