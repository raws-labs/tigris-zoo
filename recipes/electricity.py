"""Train and validate a fixed-window household electricity forecaster."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from tigris.emitters.binary.reader import read_binary_plan
from tigris.zoo import digest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = "372676d99d76069ae8c66ee93681e0be322f84bf"
RUNTIME = "a94952951924980eecfc1bb63ac49bd0c2c9d722"
DATASET = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
DATASET_SHA = "9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff"


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


def hourly_values(archive):
    values, dates = [], []
    current, total, count = None, 0.0, 0
    with zipfile.ZipFile(archive) as z:
        reader = csv.DictReader(io.TextIOWrapper(z.open("household_power_consumption.txt")), delimiter=";")
        for row in reader:
            day, month, year = map(int, row["Date"].split("/"))
            key = f"{year:04d}-{month:02d}-{day:02d}T{row['Time'][:2]}:00:00"
            if current != key:
                if current is not None:
                    dates.append(current)
                    values.append(total / 60 if count == 60 else float("nan"))
                current, total, count = key, 0.0, 0
            try:
                total += float(row["Global_active_power"])
                count += 1
            except ValueError:
                pass
    dates.append(current)
    values.append(total / 60 if count == 60 else float("nan"))
    if not np.all(np.diff(np.array(dates, dtype="datetime64[h]")).astype(int) == 1):
        raise ValueError("dataset timestamps are not consecutive hours")
    return np.array(values), np.array(dates)


def windows(values, start, end, step):
    origins = [i for i in range(max(start, 168), end - 24 + 1, step)
               if np.isfinite(values[i - 168:i + 24]).all()]
    x = np.array([values[i - 168:i] for i in origins])
    y = np.array([values[i:i + 24] for i in origins])
    mean = x.mean(axis=1, keepdims=True)
    scale = np.maximum(x.std(axis=1, keepdims=True), 1e-6)
    return np.c_[(x - mean) / scale, np.ones(len(x))], (y - mean) / scale, mean, scale, x, y, origins


def train(values, dates, output):
    train_end, validation_end = int(len(values) * 0.6), int(len(values) * 0.8)
    training = windows(values, 168, train_end, 1)
    validation = windows(values, train_end, validation_end, 24)
    test = windows(values, validation_end, len(values), 24)
    regularizer = np.eye(169)
    regularizer[-1, -1] = 0
    scores, best = [], None
    for penalty in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        weights = np.linalg.solve(training[0].T @ training[0] + penalty * regularizer,
                                  training[0].T @ training[1])
        predicted = validation[0] @ weights * validation[3] + validation[2]
        mae = float(np.abs(predicted - validation[5]).mean())
        scores.append({"penalty": penalty, "mae_kw": mae})
        if best is None or mae < best[0]:
            best = mae, penalty, weights
    _, penalty, weights = best
    weights = weights.astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["input", "weight", "bias"], ["output"], transB=1)],
        "hourly_electricity_forecast",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 168])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 24])],
        [numpy_helper.from_array(weights[:-1].T.copy(), "weight"),
         numpy_helper.from_array(weights[-1].copy(), "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output / "model.onnx")
    metrics = {
        "training_windows": len(training[0]), "validation_windows": len(validation[0]),
        "test_windows": len(test[0]), "penalty": penalty, "validation_candidates": scores,
        "train_end": dates[train_end - 1], "validation_end": dates[validation_end - 1],
        "test_start": dates[validation_end], "test_end": dates[-1],
        "daily_persistence_mae_kw": float(np.abs(test[4][:, -24:] - test[5]).mean()),
        "weekly_persistence_mae_kw": float(np.abs(test[4][:, :24] - test[5]).mean()),
        "tested_runtime_versions": ["0.9.1"], "runtime_commit": RUNTIME,
        "scope": "Chronological holdout from one household; no cross-household validation.",
    }
    return test, metrics


def build(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists; use a new build directory")
    compiler = checkout(args.compiler_source, "https://github.com/raws-labs/tigris.git", COMPILER)
    runtime = checkout(args.runtime_source, "https://github.com/raws-labs/tigris-runtime.git", RUNTIME)
    archive = args.dataset
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(DATASET, timeout=60) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        if digest(temporary) != DATASET_SHA:
            raise ValueError("dataset download checksum mismatch")
        temporary.replace(archive)
    if digest(archive) != DATASET_SHA:
        raise ValueError("dataset checksum mismatch")
    output.mkdir(parents=True)
    values, dates = hourly_values(archive)
    test, metrics = train(values, dates, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(8, int(os.environ.get("OMP_NUM_THREADS", "1")))
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output / "model.onnx"), options, providers=["CPUExecutionProvider"])
    inputs = test[0][:, :-1].astype("<f4")
    references = np.concatenate([session.run(None, {"input": row[None]})[0] for row in inputs])
    metrics["onnx_mae_kw"] = float(np.abs(references * test[3] + test[2] - test[5]).mean())
    if not metrics["onnx_mae_kw"] < min(metrics["daily_persistence_mae_kw"], metrics["weekly_persistence_mae_kw"]):
        raise ValueError("trained model does not improve on both persistence baselines")
    run(["cmake", "-S", runtime, "-B", output / "runtime", "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", output / "runtime", "--target", "tigris_runtime", "--parallel", "8"])
    recipe_hash = digest(Path(__file__))
    runner_hash = digest(ROOT / "recipes/electricity_runner.c")
    requirements_hash = digest(ROOT / "requirements-build.txt")
    try:
        recipe_commit = run(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
        if run(["git", "status", "--porcelain"], cwd=ROOT).strip():
            recipe_commit = None
    except ValueError:
        recipe_commit = None
    env = dict(os.environ, PYTHONPATH=str(compiler / "src"))
    budget = 1024
    directory = output / f"{budget // 1024}k"
    directory.mkdir()
    run([sys.executable, "-c", "from tigris.cli import main; main()", "compile",
         output / "model.onnx", "-m", str(budget), "--xip", "-o", directory / "model.tgrs"], env=env)
    plan = read_binary_plan((directory / "model.tgrs").read_bytes())
    if plan["version"] != 7 or plan["budget"] != budget:
        raise ValueError("unexpected compiled plan contract")
    generated = output / f"codegen-{budget}"
    generated.mkdir()
    run([sys.executable, "-c", "from tigris.cli import main; main()", "codegen",
         directory / "model.tgrs", "--format", "core", "-o", generated / "model.c"], env=env)
    runner = generated / "run"
    run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", runtime / "include",
         "-I", generated, ROOT / "recipes/electricity_runner.c", generated / "model.c",
         output / "runtime/libtigris_runtime.a", "-lm", "-o", runner])
    # This recipe has linear rank-2 interfaces, so no boundary transpose applies.
    predictions, peaks, slow_usage = [], [], []
    for row, reference in zip(inputs, references):
        row.tofile(output / "input.bin")
        report = run([runner, directory / "model.tgrs", output / "input.bin", output / "output.bin"])
        prediction = np.fromfile(output / "output.bin", dtype="<f4")
        np.testing.assert_allclose(prediction, reference, rtol=1e-4, atol=1e-4)
        predictions.append(prediction)
        report = json.loads(report)
        peaks.append(report["fast_peak"])
        slow_usage.append(report["slow_peak"])
    predictions = np.array(predictions)
    evaluation = dict(metrics, parity_passed=True, atol=1e-4, rtol=1e-4,
                      max_abs_error=float(np.abs(predictions - references).max()),
                      runtime_mae_kw=float(np.abs(predictions * test[3] + test[2] - test[5]).mean()),
                      measured_fast_peak_bytes=max(peaks),
                      measured_slow_bytes=max(slow_usage),
                      host_executor_workspace_bytes=report["workspace_bytes"])
    if max(peaks) > budget or abs(evaluation["runtime_mae_kw"] - metrics["onnx_mae_kw"]) > 1e-4:
        raise ValueError("runtime memory or task-quality gate failed")
    insufficient = subprocess.run([str(runner), str(directory / "model.tgrs"), str(output / "input.bin"),
                                   str(output / "output.bin"), "767"], capture_output=True)
    if insufficient.returncode != 9:
        raise ValueError("slow-arena lower-bound check failed")
    inputs[0].tofile(directory / "example-input.bin")
    references[0].astype("<f4").tofile(directory / "example-output.bin")
    origin = test[6][0]
    with (directory / "example.csv").open("w") as stream:
        stream.write("timestamp,demand_kw\n")
        for i in range(origin - 168, origin + 24):
            stream.write(f"{dates[i]},{values[i]:.8f}\n")
    (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    (directory / "license.txt").write_text(
        (ROOT / "LICENSE").read_text() + "\nDataset and example data attribution:\n"
        "Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption.\n"
        "UCI Machine Learning Repository. https://doi.org/10.24432/C58K54\n"
        "Dataset and derived example tensors/CSV: CC BY 4.0,\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        "Minute readings were averaged hourly and examples normalized.\n"
    )
    (directory / "readme.md").write_text(
        "# Hourly household electricity forecast\n\n"
        "A ridge linear model predicts 24 hours from 168 consecutive hourly mean\n"
        "active-power readings in kW. Supply nonnegative, finite values with no gaps.\n"
        "Normalize each context by its mean and population standard deviation\n"
        "(scale floor 1e-6), then supply a little-endian float32 tensor [1, 168].\n"
        "The output is float32 [1, 24]. Multiply by that context's scale and add\n"
        "its mean to recover kW. Predictions are not clipped.\n\n"
        "Training/validation/test are chronological 60/20/20 splits. Ridge penalty\n"
        "selection uses validation only. Training windows advance hourly; held-out\n"
        "origins advance daily. Hours with incomplete minute data are excluded.\n"
        "Past context may precede a split, but target horizons never cross its end.\n\n"
        f"Runtime MAE: {evaluation['runtime_mae_kw']:.6f} kW on {metrics['test_windows']} daily windows.\n"
        f"Previous-day persistence: {metrics['daily_persistence_mae_kw']:.6f} kW; "
        f"previous-week: {metrics['weekly_persistence_mae_kw']:.6f} kW.\n"
        "This evaluation covers one household; transfer to another meter is untested.\n\n"
        "The first complete holdout window supplies example.csv: 168 context rows\n"
        "followed by 24 target rows. The binary example files contain normalized\n"
        "model I/O, selected without consulting prediction error.\n\n"
        "The plan uses float32 reference kernels and weights read in place from\n"
        "flash. The 768-byte slow arena includes input and output storage. Arena\n"
        "requirements exclude external I/O staging, stack, runtime structures,\n"
        "and executor workspace. The supplied runtime release was tested on the\n"
        "host; no hardware latency measurements are claimed. Runtime 0.9.1 was tested.\n"
        "No upper compatibility bound is known. Current constraints and validation\n"
        "history are preserved in download.json when this artifact is downloaded.\n"
    )
    build_hash = hashlib.sha256((recipe_hash + runner_hash + requirements_hash + digest(directory / "model.tgrs") + COMPILER + RUNTIME).encode()).hexdigest()[:16]
    metadata = {
        "format_version": 1, "id": f"electricity-hourly-f32-s7-{budget // 1024}k-{build_hash}",
        "model": "electricity-hourly", "category": "time-series-forecasting", "schema": 7,
        "quantization": "float32", "backends": ["reference"],
        "runtime": {"min": "0.9.1", "max": None},
        "compatibility_note": "Schema 7 with float32 Gemm on the reference backend; baseline runtime 0.9.1, no known upper cutoff.",
        "memory": {"fast_bytes": budget, "slow_bytes": 768, "flash_bytes": (directory / "model.tgrs").stat().st_size},
        "inputs": [{"name": "input", "shape": [1, 168], "dtype": "float32"}],
        "outputs": [{"name": "output", "shape": [1, 24], "dtype": "float32"}],
        "license": "apache-2.0 AND cc-by-4.0",
        "compiler": {"version": "0.8.0", "commit": COMPILER},
        "source": {"dataset": DATASET, "dataset_sha256": DATASET_SHA,
                   "onnx_sha256": digest(output / "model.onnx"), "recipe": "recipes/electricity.py",
                   "recipe_sha256": recipe_hash, "recipe_commit": recipe_commit,
                   "recipe_files": {"recipes/electricity_runner.c": runner_hash,
                                    "requirements-build.txt": requirements_hash},
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
    parser.add_argument("--dataset", type=Path, default=ROOT / ".build/electricity.zip")
    parser.add_argument("--compiler-source", type=Path, default=ROOT / ".build/compiler")
    parser.add_argument("--runtime-source", type=Path, default=ROOT / ".build/runtime")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, AssertionError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
