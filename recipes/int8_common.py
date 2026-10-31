"""Checkouts, pinned downloads, the int8 runner and TFLite Micro references for the int8 recipes."""

from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

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


def fetch(url, path, sha, mirrors=()):
    """Download a pinned file once; each mirror is tried in turn when a source
    fails or serves other bytes."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        errors = []
        for source in (url, *mirrors):
            request = urllib.request.Request(source, headers={"User-Agent": "tigris-zoo-recipe"})
            print(f"fetching {source}", file=sys.stderr, flush=True)
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
            except OSError as exc:
                print(f"  failed after {time.monotonic() - started:.0f} s: {exc}", file=sys.stderr, flush=True)
                errors.append(f"{source}: {exc}")
                continue
            print(f"  {temporary.stat().st_size} bytes in {time.monotonic() - started:.0f} s",
                  file=sys.stderr, flush=True)
            if digest(temporary) == sha:
                temporary.replace(path)
                break
            errors.append(f"{source}: download checksum mismatch")
        else:
            temporary.unlink(missing_ok=True)
            raise ValueError("no source served the pinned file:\n" + "\n".join(errors))
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


def slow_arena_gate(executable, plan, workdir, slow_peak):
    """The measured slow-arena peak is the minimum the plan needs: a slow arena one
    byte smaller is refused, and a plan that used no slow memory runs without a slow
    arena and writes the same outputs. Raises on failure."""
    inputs, outputs = workdir / "inputs.bin", workdir / "outputs.bin"
    if slow_peak:
        refused = subprocess.run([str(executable), str(plan), str(inputs), str(workdir / "short.bin"),
                                  str(slow_peak - 1)], capture_output=True)
        if refused.returncode not in (8, 9):
            raise ValueError("slow-arena lower-bound check failed")
        return
    without = subprocess.run([str(executable), str(plan), str(inputs), str(workdir / "no-slow.bin"), "0"],
                             capture_output=True)
    if without.returncode or (workdir / "no-slow.bin").read_bytes() != outputs.read_bytes():
        raise ValueError("plan with no slow-arena use failed or changed output without a slow arena")


def example_image(path, size):
    """A photograph center-cropped to a square and box-filtered to size x size RGB."""
    from PIL import Image
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    square = image.crop((left, top, left + side, top + side))
    return np.asarray(square.resize((size, size), Image.Resampling.BOX), dtype=np.uint8)


def image_references(model_path, images):
    """TFLite Micro's int8 scores for uint8 NHWC images whose int8 input is pixel - 128."""
    from tflite_micro.python.tflite_micro import runtime as micro
    interpreter = micro.Interpreter.from_file(str(model_path), arena_size=1024 * 1024)
    outputs = []
    for image in images:
        interpreter.set_input((image.astype(np.int16) - 128).astype(np.int8)[None], 0)
        interpreter.invoke()
        outputs.append(interpreter.get_output(0).reshape(-1).copy())
    return np.array(outputs, dtype=np.int8)


def micro_version():
    from importlib.metadata import version
    return version("tflite-micro")
