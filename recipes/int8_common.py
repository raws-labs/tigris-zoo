"""Source checkouts, pinned downloads and the int8 runner shared by the int8 recipes."""

from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

from tigris.zoo import digest

ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    result = subprocess.run([str(arg) for arg in args], text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise ValueError(f"command failed ({result.returncode}): {args[0]}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def checkout(path, repository, revision):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", repository, path])
        run(["git", "-C", path, "checkout", "--detach", revision])
    if run(["git", "-C", path, "rev-parse", "HEAD"]).strip() != revision:
        raise ValueError(f"source checkout must be at {revision}")
    if run(["git", "-C", path, "status", "--porcelain"]).strip():
        raise ValueError("source checkout must be clean")
    return path.resolve()


def fetch(url, path, sha):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        request = urllib.request.Request(url, headers={"User-Agent": "tigris-zoo-recipe"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        if digest(temporary) != sha:
            raise ValueError(f"download checksum mismatch: {url}")
        temporary.replace(path)
    if digest(path) != sha:
        raise ValueError(f"checksum mismatch: {path}")
    return path


def recipe_commit():
    """The checked-out recipe commit, or None when the tree has local changes."""
    try:
        commit = run(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
        return None if run(["git", "status", "--porcelain"], cwd=ROOT).strip() else commit
    except ValueError:
        return None


def runner(runtime, output, plan, generated, env=None):
    """Build the runtime library and the int8 runner linked with the plan's generated core."""
    run(["cmake", "-S", runtime, "-B", output / "runtime", "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", output / "runtime", "--target", "tigris_runtime", "--parallel", "8"])
    generated.mkdir()
    run([sys.executable, "-c", "from tigris.cli import main; main()", "codegen",
         plan, "--format", "core", "-o", generated / "model.c"], env=env)
    executable = generated / "run"
    run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", runtime / "include",
         "-I", generated, ROOT / "recipes/int8_runner.c", generated / "model.c",
         output / "runtime/libtigris_runtime.a", "-lm", "-o", executable])
    return executable
