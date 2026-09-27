"""Compile and validate the UCI HAR activity-recognition 1D CNN as an int8 plan."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from tigris.emitters.binary.reader import read_binary_plan
from tigris.zoo import digest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from har_data import (CHANNELS, DATASET, DATASET_SHA, LABELS, baseline_accuracy, load,  # noqa: E402
                      normalization, normalize)

ROOT = Path(__file__).resolve().parents[1]
COMPILER = "908b0fd0a8687f45e20594ba95977aa8695c26ff"
COMPILER_VERSION = "0.11.3"
RUNTIME = "96ed57e1d9122db779f9e8d388a81098e0e8f27c"
RUNTIME_VERSION = "0.11.3"
# The int8 QDQ model recipes/har_train.py produced; training is not
# bit-reproducible across CPUs, so the recipe pins its output.
MODEL = "https://huggingface.co/raws-labs/tigris-zoo/resolve/main/sources/har/har_int8.onnx"
MODEL_SHA = "7f9ce3c0e5db9f9d34d63f16d1d562ba0b7a722a8e9d60fbd2d25a30eaad484b"
BUDGET = 4 * 1024
# ONNX Runtime requantizes each layer in float, the runtime with integer
# multipliers; the two may round one step apart.
MAX_LSB = 1


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
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        if digest(temporary) != sha:
            raise ValueError(f"download checksum mismatch: {url}")
        temporary.replace(path)
    if digest(path) != sha:
        raise ValueError(f"checksum mismatch: {path}")
    return path


def input_quantization(model):
    """Scale and zero point of the QuantizeLinear that reads the model input."""
    initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    node = next(n for n in model.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == "input")
    return float(initializers[node.input[1]]), int(initializers[node.input[2]])


def quantize(windows, scale, zero_point):
    """The runtime's float32 input conversion: divide in float32, round half
    away from zero, saturate."""
    scaled = windows / np.float32(scale)
    return np.clip(np.trunc(scaled + np.copysign(np.float32(0.5), scaled)) + zero_point,
                   -128, 127).astype(np.int8)


def model(output, data):
    """The pinned int8 model, the normalized test windows and their references."""
    source = fetch(MODEL, data / "har_int8.onnx", MODEL_SHA)
    archive = fetch(DATASET, data / "har.zip", DATASET_SHA)
    shutil.copyfile(source, output / "model.onnx")
    graph = onnx.load(source)
    splits = load(archive)
    mean, std = normalization(splits["train"][0])
    props = {p.key: p.value for p in graph.metadata_props}
    if (json.loads(props["normalization_mean"]), json.loads(props["normalization_std"])) != (mean.tolist(), std.tolist()):
        raise ValueError("the pinned model was trained with different normalization constants")
    windows, labels, _ = splits["test"]
    windows = normalize(windows, mean, std)
    scale, zero_point = input_quantization(graph)
    quantized = quantize(windows, scale, zero_point)
    # ONNX Runtime quantizes ties to even, so it receives the exact values the
    # runtime's int8 input stands for and both start from identical inputs.
    exact = ((quantized.astype(np.float32) - zero_point) * np.float32(scale)).astype(np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(8, int(os.environ.get("OMP_NUM_THREADS", "1")))
    session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    reference = np.concatenate([session.run(None, {"input": row[None]})[0] for row in exact])
    metrics = {
        "test_windows": len(labels), "labels": LABELS, "channels": CHANNELS,
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "reference": "ONNX Runtime CPU execution of the int8 QDQ model", "onnxruntime": ort.__version__,
        "reference_accuracy": float((reference.argmax(axis=1) == labels).mean()),
        "baseline": "ridge classifier on per-channel mean and standard deviation",
        "baseline_accuracy": baseline_accuracy(splits),
        "tested_runtime_versions": [RUNTIME_VERSION], "runtime_commit": RUNTIME,
        "scope": "UCI HAR test subjects as distributed; one phone model worn at the waist.",
    }
    return windows, labels, quantized, reference, metrics


def runner(runtime, output, plan, generated, env=None):
    """Build the runtime library and a runner linked with the plan's generated core."""
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


def plan_layout(plan_path, quantized):
    """The int8 windows in the plan's stored input layout."""
    plan = read_binary_plan(plan_path.read_bytes())
    shape = list(plan["tensors"][plan["model_inputs"][0]]["shape"])
    if shape == [1, 128, 6]:
        return quantized.transpose(0, 2, 1).copy(), shape
    if shape == [1, 6, 128]:
        return quantized, shape
    raise ValueError(f"unexpected plan input shape {shape}")


def runtime_gates(executable, plan, quantized, reference, labels, metrics, fast_bytes, workdir):
    """The acceptance gates for a plan on one runtime: every logit within one
    LSB of ONNX Runtime on every window, higher accuracy than the baseline
    classifier, the fast budget, and refusal of a slow arena one byte short.
    Raises on any failure."""
    stored, _ = plan_layout(plan, quantized)
    stored.tofile(workdir / "inputs.bin")
    report = json.loads(run([executable, plan, workdir / "inputs.bin", workdir / "outputs.bin"]))
    scores = np.fromfile(workdir / "outputs.bin", dtype=np.int8).reshape(reference.shape).astype(np.int32)
    if report["samples"] != len(reference):
        raise ValueError("runner did not process every window")
    graph = onnx.load(workdir / "model.onnx").graph
    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    last = next(n for n in graph.node if n.output[0] == "output")
    out_scale, out_zero = float(initializers[last.input[1]]), int(initializers[last.input[2]])
    expected = np.round(reference / out_scale).astype(np.int32) + out_zero
    lsb = np.abs(scores - expected)
    agreement = int((scores.argmax(axis=1) == expected.argmax(axis=1)).sum())
    accuracy = float((scores.argmax(axis=1) == labels).mean())
    evaluation = dict(metrics, parity_passed=bool(lsb.max() <= MAX_LSB), max_lsb_allowed=MAX_LSB,
                      top1_agreement=agreement,
                      windows_differing=int((lsb.max(axis=1) > 0).sum()), max_abs_lsb=int(lsb.max()),
                      runtime_accuracy=accuracy,
                      measured_fast_peak_bytes=report["fast_peak"],
                      measured_slow_bytes=report["slow_peak"],
                      host_executor_workspace_bytes=report["workspace_bytes"])
    if lsb.max() > MAX_LSB:
        raise ValueError(f"runtime logits differ from ONNX Runtime by up to {int(lsb.max())} LSB")
    if accuracy <= metrics["baseline_accuracy"] or report["fast_peak"] > fast_bytes:
        raise ValueError("runtime accuracy or memory gate failed")
    insufficient = subprocess.run([str(executable), str(plan), str(workdir / "inputs.bin"),
                                   str(workdir / "short.bin"), str(report["slow_peak"] - 1)],
                                  capture_output=True)
    if insufficient.returncode not in (8, 9):
        raise ValueError("slow-arena lower-bound check failed")
    return evaluation


def revalidate(plan, manifest, runtime, output, data):
    """Rerun this recipe's runtime gates on published plan bytes with another runtime
    checkout, using the installed compiler's code generator. Returns the evaluation."""
    if digest(plan) != next(item["sha256"] for item in manifest["files"] if item["path"] == "model.tgrs"):
        raise ValueError("plan bytes differ from the published manifest")
    source = manifest["source"]
    if (source["model_sha256"], source["dataset_sha256"]) != (MODEL_SHA, DATASET_SHA):
        raise ValueError("artifact was built from a different model or dataset")
    output.mkdir(parents=True)
    _, labels, quantized, reference, metrics = model(output, data)
    executable = runner(runtime, output, plan, output / "codegen")
    return runtime_gates(executable, plan, quantized, reference, labels, metrics,
                         manifest["memory"]["fast_bytes"], output)


def build(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists; use a new build directory")
    compiler = checkout(args.compiler_source, "https://github.com/raws-labs/tigris.git", COMPILER)
    runtime = checkout(args.runtime_source, "https://github.com/raws-labs/tigris-runtime.git", RUNTIME)
    output.mkdir(parents=True)
    windows, labels, quantized, reference, metrics = model(output, args.data.resolve())
    if metrics["reference_accuracy"] <= metrics["baseline_accuracy"]:
        raise ValueError("the pinned model does not beat the baseline classifier")
    recipe_files = {name: digest(ROOT / name) for name in
                    ("recipes/int8_runner.c", "recipes/har_data.py", "recipes/har_train.py",
                     "requirements-build.txt")}
    recipe_hash = digest(Path(__file__))
    try:
        recipe_commit = run(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
        if run(["git", "status", "--porcelain"], cwd=ROOT).strip():
            recipe_commit = None
    except ValueError:
        recipe_commit = None
    env = dict(os.environ, PYTHONPATH=str(compiler / "src"))
    directory = output / f"{BUDGET // 1024}k"
    directory.mkdir()
    run([sys.executable, "-c", "from tigris.cli import main; main()", "compile",
         output / "model.onnx", "-m", str(BUDGET), "--xip", "-o", directory / "model.tgrs"], env=env)
    plan = read_binary_plan((directory / "model.tgrs").read_bytes())
    if plan["version"] != 9 or plan["budget"] != BUDGET:
        raise ValueError("unexpected compiled plan contract")
    interface = [[plan["tensors"][i]["name"] for i in plan[key]] for key in ("model_inputs", "model_outputs")]
    if interface != [["input"], ["output"]]:
        raise ValueError(f"unexpected plan interface {interface}")
    executable = runner(runtime, output, directory / "model.tgrs", output / f"codegen-{BUDGET}", env=env)
    evaluation = runtime_gates(executable, directory / "model.tgrs", quantized, reference, labels, metrics,
                               BUDGET, output)

    # The first test window, chosen without consulting any prediction.
    first, shape = plan_layout(directory / "model.tgrs", windows[:1])
    first.astype("<f4").tofile(directory / "example-input.bin")
    reference[0].astype("<f4").tofile(directory / "example-output.bin")
    (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    (directory / "license.txt").write_text(
        (ROOT / "LICENSE").read_text() + "\nDataset and example data attribution:\n"
        "Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L. & Parra, X. (2013). Human\n"
        "Activity Recognition Using Smartphones. UCI Machine Learning Repository.\n"
        "https://doi.org/10.24432/C54S4K\n"
        "Dataset and the model trained on it, with the derived example tensors: CC BY 4.0,\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
    )
    mean = ", ".join(f"{v:.6g}" for v in metrics["normalization_mean"])
    std = ", ".join(f"{v:.6g}" for v in metrics["normalization_std"])
    (directory / "readme.md").write_text(
        "# Human activity recognition (1D CNN, int8)\n\n"
        "A three-layer 1D CNN classifies 2.56 s of waist-worn smartphone motion into\n"
        f"6 activities: {', '.join(LABELS)}.\n\n"
        f"Input: float32 {shape} in the plan's stored layout, 128 time steps of 6\n"
        "channels at 50 Hz: total acceleration x, y, z in g and angular velocity\n"
        "x, y, z in rad/s, as in the UCI HAR inertial signals. Normalize each channel\n"
        f"by subtracting its mean ({mean}) and dividing by its standard\n"
        f"deviation ({std}). The runtime quantizes the input at the boundary.\n"
        "Output: float32 [1, 6] class logits from an int8 layer; the largest is the\n"
        "predicted activity.\n\n"
        "The model was trained once with recipes/har_train.py on 16 of the 21 training\n"
        "subjects, selected on the other 5, and quantized with ONNX Runtime static\n"
        "QDQ quantization. Training is not bit-reproducible across CPUs, so the\n"
        "recipe pins that int8 model by SHA-256.\n\n"
        f"On the {metrics['test_windows']} windows of the 9 test subjects every runtime logit is\n"
        f"within {MAX_LSB} LSB of ONNX Runtime's, and top-1 accuracy is\n"
        f"{evaluation['runtime_accuracy']:.4f}. The recipe requires it to beat a ridge classifier\n"
        f"on each channel's mean and standard deviation, which reaches {metrics['baseline_accuracy']:.4f}.\n"
        "The largest error group confuses sitting with standing. ONNX Runtime requantizes in\n"
        "float, the runtime with integer multipliers, so logits differ by one LSB on\n"
        f"{evaluation['windows_differing']} windows. Where the top two logits are within one LSB, the\n"
        f"predicted class can differ; it matches on {evaluation['top1_agreement']} windows.\n\n"
        "example-input.bin holds the first test window and example-output.bin the\n"
        "ONNX Runtime logits for it.\n\n"
        f"The plan uses int8 reference kernels with weights read in place from flash\n"
        f"and a {BUDGET // 1024} KiB fast arena; the {evaluation['measured_slow_bytes']}-byte slow arena holds\n"
        "the model input and output. Arena requirements exclude stack, runtime\n"
        "structures, and executor workspace. Tested on the host with runtime\n"
        f"{RUNTIME_VERSION}; no hardware latency measurements are claimed. Runtimes accept\n"
        "the plan from 0.10.0. Current constraints and validation history are\n"
        "preserved in download.json when this artifact is downloaded.\n"
    )
    build_hash = hashlib.sha256((recipe_hash + "".join(recipe_files.values()) + digest(directory / "model.tgrs")
                                 + COMPILER + RUNTIME).encode()).hexdigest()[:16]
    metadata = {
        "format_version": 1, "id": f"har-cnn1d-s8-s9-{BUDGET // 1024}k-{build_hash}",
        "model": "har-cnn1d", "category": "activity-recognition", "schema": 9,
        "quantization": "int8", "backends": ["reference"],
        "runtime": {"min": "0.10.0", "max": None},
        "compatibility_note": "Schema 9 int8 plan, which runtimes accept from 0.10.0; no known upper cutoff.",
        "memory": {"fast_bytes": BUDGET, "slow_bytes": evaluation["measured_slow_bytes"],
                   "flash_bytes": (directory / "model.tgrs").stat().st_size},
        "inputs": [{"name": "input", "shape": shape, "dtype": "float32"}],
        "outputs": [{"name": "output", "shape": [1, len(LABELS)], "dtype": "float32"}],
        "license": "apache-2.0 AND cc-by-4.0",
        "compiler": {"version": COMPILER_VERSION, "commit": COMPILER},
        "source": {"model": MODEL, "model_sha256": MODEL_SHA, "dataset": DATASET, "dataset_sha256": DATASET_SHA,
                   "onnx_sha256": digest(output / "model.onnx"), "recipe": "recipes/har.py",
                   "recipe_sha256": recipe_hash, "recipe_commit": recipe_commit, "recipe_files": recipe_files,
                   "numpy": np.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "evaluation": evaluation,
        "files": [{"path": p.name, "size": p.stat().st_size, "sha256": digest(p)}
                  for p in sorted(directory.iterdir())],
    }
    (directory / "artifact.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"id": metadata["id"], "evaluation": evaluation}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / ".build", help="Download cache for model and dataset")
    parser.add_argument("--compiler-source", type=Path, default=ROOT / ".build/compiler-har")
    parser.add_argument("--runtime-source", type=Path, default=ROOT / ".build/runtime-har")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, AssertionError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
