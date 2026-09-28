"""Compile and validate the MLPerf Tiny image-classification ResNet-8 as an int8 plan."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile

import numpy as np
import onnx
from PIL import Image

from tigris.emitters.binary.reader import read_binary_plan
from tigris.zoo import digest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from int8_common import (ROOT, checkout, example_image, fetch, image_references, micro_version,  # noqa: E402
                         recipe_commit, run, runner)
from tflite_qdq import convert  # noqa: E402

COMPILER = "908b0fd0a8687f45e20594ba95977aa8695c26ff"
COMPILER_VERSION = "0.11.3"
RUNTIME = "96ed57e1d9122db779f9e8d388a81098e0e8f27c"
RUNTIME_VERSION = "0.11.3"
MODEL = ("https://raw.githubusercontent.com/mlcommons/tiny/4addd0fa08d216e20637637874e084895f289da4/"
         "benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite")
MODEL_SHA = "3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415"
DATASET = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATASET_SHA = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
# The example is a public-domain photograph, not a CIFAR-10 image: the test set
# is used for evaluation only and not redistributed.
EXAMPLE = ("https://upload.wikimedia.org/wikipedia/commons/f/f1/"
           "OV-10A_%28NASA-718%29_Airplane_in_flight_%28ARC-1973-A73-2003%29.jpg")
EXAMPLE_SHA = "aa0438c1e2cf2e155083d6b3757a1d3e3a58b9b8edba4692333e45162c23bdd3"
LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# MLPerf Tiny's top-1 quality target for this benchmark.
ACCURACY_FLOOR = 0.85
BUDGET = 48 * 1024
INPUT_SHAPE = [1, 32, 32, 3]


def test_images(archive):
    """CIFAR-10 test images as uint8 NHWC, with their labels."""
    with tarfile.open(archive, "r:gz") as tar:
        batch = pickle.loads(tar.extractfile("cifar-10-batches-py/test_batch").read(), encoding="bytes")
    return batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy(), np.array(batch[b"labels"])








def model(output, data):
    """The source model, its ONNX conversion, and the test set with references."""
    tflite_path = fetch(MODEL, data / "resnet8.tflite", MODEL_SHA)
    archive = fetch(DATASET, data / "cifar-10-python.tar.gz", DATASET_SHA)
    onnx.save(convert(tflite_path.read_bytes(), "resnet8"), output / "model.onnx")
    images, labels = test_images(archive)
    reference = image_references(tflite_path, images)
    metrics = {
        "test_images": len(labels), "labels": LABELS,
        "reference": "TFLite Micro reference kernels", "reference_package": f"tflite-micro {micro_version()}",
        "reference_accuracy": float((reference.astype(np.int32).argmax(axis=1) == labels).mean()),
        "accuracy_floor": ACCURACY_FLOOR,
        "tested_runtime_versions": [RUNTIME_VERSION], "runtime_commit": RUNTIME,
        "scope": "CIFAR-10 test set; 32 x 32 images of ten object classes.",
    }
    return images, labels, reference, tflite_path, metrics


def runtime_gates(executable, plan, images, reference, labels, metrics, fast_bytes, workdir):
    """The acceptance gates for a plan on one runtime: int8 scores identical to
    TFLite Micro on every image, accuracy at or above the floor, the fast budget,
    and refusal of a slow arena one byte short. Raises on any failure."""
    (images.astype(np.int16) - 128).astype(np.int8).tofile(workdir / "inputs.bin")
    report = json.loads(run([executable, plan, workdir / "inputs.bin", workdir / "outputs.bin"]))
    scores = np.fromfile(workdir / "outputs.bin", dtype=np.int8).reshape(reference.shape)
    if report["samples"] != len(reference):
        raise ValueError("runner did not process every image")
    differing = int((scores != reference).any(axis=1).sum())
    accuracy = float((scores.astype(np.int32).argmax(axis=1) == labels).mean())
    evaluation = dict(metrics, parity_passed=differing == 0, images_differing=differing,
                      max_abs_lsb=int(np.abs(scores.astype(np.int32) - reference).max()),
                      runtime_accuracy=accuracy,
                      measured_fast_peak_bytes=report["fast_peak"],
                      measured_slow_bytes=report["slow_peak"],
                      host_executor_workspace_bytes=report["workspace_bytes"])
    if differing:
        raise ValueError(f"runtime scores differ from TFLite Micro on {differing} images")
    if accuracy < ACCURACY_FLOOR or report["fast_peak"] > fast_bytes:
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
    images, labels, reference, _, metrics = model(output, data)
    if digest(output / "model.onnx") != source["onnx_sha256"]:
        raise ValueError("conversion did not reproduce the published model")
    executable = runner(runtime, output, plan, output / "codegen")
    return runtime_gates(executable, plan, images, reference, labels, metrics,
                         manifest["memory"]["fast_bytes"], output)


def build(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists; use a new build directory")
    compiler = checkout(args.compiler_source, "https://github.com/raws-labs/tigris.git", COMPILER)
    runtime = checkout(args.runtime_source, "https://github.com/raws-labs/tigris-runtime.git", RUNTIME)
    output.mkdir(parents=True)
    data = args.data.resolve()
    images, labels, reference, tflite_path, metrics = model(output, data)
    if metrics["reference_accuracy"] < ACCURACY_FLOOR:
        raise ValueError("the source model misses the accuracy floor on the test set")
    photo = fetch(EXAMPLE, data / "resnet8-example.jpg", EXAMPLE_SHA)
    recipe_files = {name: digest(ROOT / name) for name in
                    ("recipes/int8_common.py", "recipes/int8_runner.c", "recipes/tflite_qdq.py",
                     "requirements-build.txt")}
    recipe_hash = digest(Path(__file__))
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
    evaluation = runtime_gates(executable, directory / "model.tgrs", images, reference, labels, metrics,
                               BUDGET, output)

    example = example_image(photo, 32)
    example.astype("<f4").reshape(INPUT_SHAPE).tofile(directory / "example-input.bin")
    example_scores = image_references(tflite_path, example[None])[0]
    ((example_scores.astype(np.float32) + 128) / 256).astype("<f4").tofile(directory / "example-output.bin")
    Image.fromarray(example).save(directory / "example.png", optimize=False)
    (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    (directory / "license.txt").write_text(
        (ROOT / "LICENSE").read_text() + "\nModel attribution:\n"
        "MLCommons MLPerf Tiny image-classification reference model\n"
        "(pretrainedResnet_quant.tflite), https://github.com/mlcommons/tiny,\n"
        "Apache License 2.0.\n"
        "\nExample image:\n"
        "OV-10A (NASA-718) airplane in flight, NASA Ames Research Center, 1973,\n"
        "ARC-1973-A73-2003. Public domain as a work of the United States government.\n"
        f"{EXAMPLE}\n"
        "example.png and example-input.bin are the photograph center-cropped and\n"
        "reduced to 32 x 32 pixels.\n"
        "\nEvaluation data: CIFAR-10 (Krizhevsky, 2009). The test set is used for\n"
        "evaluation only; no CIFAR-10 image is included.\n"
    )
    predicted = LABELS[int(example_scores.astype(np.int32).argmax())]
    (directory / "readme.md").write_text(
        "# Image classification (ResNet-8, int8)\n\n"
        "The MLPerf Tiny image-classification reference model, a ResNet with three\n"
        "residual stages, classifies a 32 x 32 RGB image into the ten CIFAR-10\n"
        f"classes: {', '.join(LABELS)}.\n\n"
        "Input: float32 [1, 32, 32, 3], height by width by RGB, raw pixel values\n"
        "0 to 255 with no normalization. The runtime quantizes them at the\n"
        "boundary to pixel - 128. Output: float32 [1, 10] class probabilities from\n"
        "an int8 Softmax, in steps of 1/256.\n\n"
        f"On the {metrics['test_images']} CIFAR-10 test images the runtime's int8 scores equal\n"
        f"TFLite Micro's on every image, and top-1 accuracy is {evaluation['runtime_accuracy']:.4f}.\n"
        f"The recipe requires {ACCURACY_FLOOR:.2f}, MLPerf Tiny's quality target for this benchmark.\n\n"
        "example.png is a public-domain photograph of an airplane reduced to 32 x 32;\n"
        "it is not from CIFAR-10. example-input.bin holds its pixels and\n"
        "example-output.bin the TFLite Micro scores dequantized to float32, which\n"
        f"`tigris run` reproduces exactly. The model predicts {predicted} for it.\n\n"
        f"The plan uses int8 reference kernels with weights read in place from flash.\n"
        f"The {BUDGET // 1024} KiB fast arena holds all activations; the "
        f"{evaluation['measured_slow_bytes']}-byte slow arena holds\n"
        "the model input and output. Smaller fast budgets tile the model but move\n"
        "48 KiB of activations into the slow arena, more RAM in total, so only this\n"
        "variant is built. Arena requirements exclude stack, runtime structures,\n"
        f"and executor workspace. Tested on the host with runtime {RUNTIME_VERSION}; no\n"
        "hardware latency measurements are claimed. Runtime 0.11.1 or newer is\n"
        "required: older runtimes refuse the plan's average-pool rounding attribute.\n"
        "Current constraints and validation history are preserved in download.json\n"
        "when this artifact is downloaded.\n"
    )
    build_hash = hashlib.sha256((recipe_hash + "".join(recipe_files.values()) + digest(directory / "model.tgrs")
                                 + COMPILER + RUNTIME).encode()).hexdigest()[:16]
    metadata = {
        "format_version": 1, "id": f"resnet8-s8-s9-{BUDGET // 1024}k-{build_hash}",
        "model": "resnet8", "category": "image-classification", "schema": 9,
        "quantization": "int8", "backends": ["reference"],
        "runtime": {"min": "0.11.1", "max": None},
        "compatibility_note": ("Schema 9 int8 plan with the average-pool rounding attribute, which runtimes "
                               "before 0.11.1 refuse; no known upper cutoff."),
        "memory": {"fast_bytes": BUDGET, "slow_bytes": evaluation["measured_slow_bytes"],
                   "flash_bytes": (directory / "model.tgrs").stat().st_size},
        "inputs": [{"name": "input", "shape": INPUT_SHAPE, "dtype": "float32"}],
        "outputs": [{"name": "output", "shape": [1, len(LABELS)], "dtype": "float32"}],
        "license": "apache-2.0",
        "compiler": {"version": COMPILER_VERSION, "commit": COMPILER},
        "source": {"model": MODEL, "model_sha256": MODEL_SHA, "dataset": DATASET, "dataset_sha256": DATASET_SHA,
                   "example": EXAMPLE, "example_sha256": EXAMPLE_SHA,
                   "onnx_sha256": digest(output / "model.onnx"), "recipe": "recipes/resnet8.py",
                   "recipe_sha256": recipe_hash, "recipe_commit": recipe_commit(), "recipe_files": recipe_files,
                   "numpy": np.__version__, "onnx": onnx.__version__, "tflite-micro": micro_version()},
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
    parser.add_argument("--compiler-source", type=Path, default=ROOT / ".build/compiler-resnet8")
    parser.add_argument("--runtime-source", type=Path, default=ROOT / ".build/runtime-resnet8")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, AssertionError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
