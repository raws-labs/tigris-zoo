"""Compile and validate the MLPerf Tiny visual wake words person detector as int8 plans."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
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
         "benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite")
MODEL_SHA = "597a384c8c2c8a1276f04702f25013b7838f2f814f1ca7c174d295b73e3d6b7b"
DATASET = "https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz"
DATASET_SHA = "f8746b9e44f8a7a4293f73be9ba6e8da9239fe69798d42364aae62b915cfab58"
# The example is a public-domain photograph; the COCO-derived images are used
# for evaluation only and not redistributed.
EXAMPLE = "https://upload.wikimedia.org/wikipedia/commons/d/d9/STS-135_Astrovan_pre-flight_photo.jpg"
EXAMPLE_SHA = "a571c353f769a7e41f891e1cc81ddab807624329d5143e4e52505c71ce6a0aa3"
LABELS = ["non_person", "person"]
# MLPerf Tiny's top-1 quality target for this benchmark.
ACCURACY_FLOOR = 0.80
# The least total RAM (tiled) and the smallest single-stage plan.
BUDGETS = (10 * 1024, 64 * 1024)
INPUT_SHAPE = [1, 96, 96, 3]


def test_images(archive):
    """The images MLPerf Tiny's training script holds out: its Keras generator
    uses validation_split=0.1, which takes the first tenth of each class
    directory in name order for validation and trains on the rest."""
    with tarfile.open(archive, "r:gz") as tar:
        data = {m.name: tar.extractfile(m).read() for m in tar if m.isfile() and m.name.endswith(".jpg")}
    images, labels = [], []
    for label, name in enumerate(LABELS):
        files = sorted(path for path in data if path.split("/")[-2] == name)
        for path in files[:int(0.1 * len(files))]:
            image = Image.open(io.BytesIO(data[path])).convert("RGB")
            if image.size != (96, 96):
                raise ValueError(f"unexpected image size {image.size}: {path}")
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(label)
    return np.array(images), np.array(labels)


def model(output, data):
    """The source model, its ONNX conversion, and the test images with references."""
    tflite_path = fetch(MODEL, data / "vww_96_int8.tflite", MODEL_SHA)
    archive = fetch(DATASET, data / "vw_coco2014_96.tar.gz", DATASET_SHA)
    onnx.save(convert(tflite_path.read_bytes(), "vww"), output / "model.onnx")
    images, labels = test_images(archive)
    reference = image_references(tflite_path, images)
    metrics = {
        "test_images": len(labels), "labels": LABELS,
        "reference": "TFLite Micro reference kernels", "reference_package": f"tflite-micro {micro_version()}",
        "reference_accuracy": float((reference.astype(np.int32).argmax(axis=1) == labels).mean()),
        "accuracy_floor": ACCURACY_FLOOR,
        "tested_runtime_versions": [RUNTIME_VERSION], "runtime_commit": RUNTIME,
        "scope": ("The held-out tenth of the vw_coco2014_96 archive MLPerf Tiny trained on: "
                  "COCO 2014 images labeled person when a person box covers at least 0.5% of the image."),
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


def readme(budget, evaluation, metrics, predicted, stages):
    tiling = (f"The plan runs in {stages} stages so that each fits the {budget // 1024} KiB fast arena; the slow\n"
              "arena holds the input image and the tensors passed between stages. Of the\n"
              "fast budgets from 8 to 192 KiB compiled for this recipe, this one needs the\n"
              "least fast plus slow RAM.\n\n" if stages > 1 else
              f"The plan runs in one stage: the {budget // 1024} KiB fast arena holds all activations and\n"
              "the slow arena holds the model input and output.\n\n")
    return (
        "# Person detection (MobileNetV1 0.25, int8)\n\n"
        "The MLPerf Tiny visual wake words reference model, a MobileNetV1 with width\n"
        "multiplier 0.25, decides whether a 96 x 96 RGB image contains a person.\n\n"
        "Input: float32 [1, 96, 96, 3], height by width by RGB, pixel values divided\n"
        "by 255. The runtime quantizes them at the boundary to pixel - 128. Output:\n"
        f"float32 [1, 2] probabilities for {LABELS[0]} and {LABELS[1]} from an int8 Softmax,\n"
        "in steps of 1/256.\n\n"
        f"The test set is the {metrics['test_images']} images MLPerf Tiny's training script holds\n"
        "out of the vw_coco2014_96 archive: the first tenth of each class in name\n"
        "order, which its Keras generator reserves for validation and never trains\n"
        "on. The runtime's int8 scores equal TFLite Micro's on every image, and top-1\n"
        f"accuracy is {evaluation['runtime_accuracy']:.4f}. The recipe requires {ACCURACY_FLOOR:.2f}, MLPerf Tiny's\n"
        "quality target for this benchmark.\n\n"
        "example.png is a public-domain NASA photograph of four astronauts reduced to\n"
        "96 x 96; it is not from COCO. example-input.bin holds its pixels divided by\n"
        "255 and example-output.bin the TFLite Micro scores dequantized to float32,\n"
        f"which `tigris run` reproduces exactly. The model predicts {predicted} for it.\n\n"
        "The plan uses int8 reference kernels with weights read in place from flash.\n"
        + tiling +
        f"Measured: {evaluation['measured_fast_peak_bytes']} bytes fast, "
        f"{evaluation['measured_slow_bytes']} bytes slow. Arena requirements\n"
        "exclude stack, runtime structures, and executor workspace. Tested on the\n"
        f"host with runtime {RUNTIME_VERSION}; no hardware latency measurements are claimed.\n"
        "Runtime 0.11.1 or newer is required: older runtimes refuse the plan's\n"
        "average-pool rounding attribute. Current constraints and validation history\n"
        "are preserved in download.json when this artifact is downloaded.\n"
    )


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
    photo = fetch(EXAMPLE, data / "vww-example.jpg", EXAMPLE_SHA)
    example = example_image(photo, 96)
    example_scores = image_references(tflite_path, example[None])[0]
    predicted = LABELS[int(example_scores.astype(np.int32).argmax())]
    recipe_files = {name: digest(ROOT / name) for name in
                    ("recipes/int8_common.py", "recipes/int8_runner.c", "recipes/tflite_qdq.py",
                     "requirements-build.txt")}
    recipe_hash = digest(Path(__file__))
    env = dict(os.environ, PYTHONPATH=str(compiler / "src"))
    for budget in BUDGETS:
        directory = output / f"{budget // 1024}k"
        directory.mkdir()
        run([sys.executable, "-c", "from tigris.cli import main; main()", "compile",
            output / "model.onnx", "-m", str(budget), "--xip", "-o", directory / "model.tgrs"], env=env)
        plan = read_binary_plan((directory / "model.tgrs").read_bytes())
        if plan["version"] != 9 or plan["budget"] != budget:
            raise ValueError("unexpected compiled plan contract")
        interface = [[plan["tensors"][i]["name"] for i in plan[key]] for key in ("model_inputs", "model_outputs")]
        if interface != [["input"], ["output"]]:
            raise ValueError(f"unexpected plan interface {interface}")
        stages = len(plan["stages"])
        executable = runner(runtime, output, directory / "model.tgrs", output / f"codegen-{budget}", env=env)
        evaluation = runtime_gates(executable, directory / "model.tgrs", images, reference, labels, metrics,
                                   budget, output)
        evaluation["stages"] = stages
        (example.astype(np.float32) / np.float32(255)).astype("<f4").reshape(INPUT_SHAPE).tofile(
            directory / "example-input.bin")
        ((example_scores.astype(np.float32) + 128) / 256).astype("<f4").tofile(directory / "example-output.bin")
        Image.fromarray(example).save(directory / "example.png", optimize=False)
        (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
        (directory / "license.txt").write_text(
            (ROOT / "LICENSE").read_text() + "\nModel attribution:\n"
            "MLCommons MLPerf Tiny visual wake words reference model (vww_96_int8.tflite),\n"
            "https://github.com/mlcommons/tiny, Apache License 2.0.\n"
            "\nExample image:\n"
            "STS-135 crew walking to the Astrovan, Kennedy Space Center, 2011-07-08,\n"
            "NASA/Kim Shiflett, KSC-2011-5206. Public domain as a work of the United\n"
            f"States government. {EXAMPLE}\n"
            "example.png and example-input.bin are the photograph center-cropped and\n"
            "reduced to 96 x 96 pixels.\n"
            "\nEvaluation data: the vw_coco2014_96 archive, derived from COCO 2014\n"
            "(Lin et al., 2014). It is used for evaluation only; no COCO image is included.\n"
        )
        (directory / "readme.md").write_text(readme(budget, evaluation, metrics, predicted, stages))
        build_hash = hashlib.sha256((recipe_hash + "".join(recipe_files.values()) + digest(directory / "model.tgrs")
                                     + COMPILER + RUNTIME).encode()).hexdigest()[:16]
        metadata = {
            "format_version": 1, "id": f"vww-mobilenetv1-s8-s9-{budget // 1024}k-{build_hash}",
            "model": "vww-mobilenetv1", "category": "person-detection", "schema": 9,
            "quantization": "int8", "backends": ["reference"],
            "runtime": {"min": "0.11.1", "max": None},
            "compatibility_note": ("Schema 9 int8 plan with the average-pool rounding attribute, which runtimes "
                                   "before 0.11.1 refuse; no known upper cutoff."),
            "memory": {"fast_bytes": budget, "slow_bytes": evaluation["measured_slow_bytes"],
                       "flash_bytes": (directory / "model.tgrs").stat().st_size},
            "inputs": [{"name": "input", "shape": INPUT_SHAPE, "dtype": "float32"}],
            "outputs": [{"name": "output", "shape": [1, len(LABELS)], "dtype": "float32"}],
            "license": "apache-2.0",
            "compiler": {"version": COMPILER_VERSION, "commit": COMPILER},
            "source": {"model": MODEL, "model_sha256": MODEL_SHA, "dataset": DATASET,
                       "dataset_sha256": DATASET_SHA, "example": EXAMPLE, "example_sha256": EXAMPLE_SHA,
                       "onnx_sha256": digest(output / "model.onnx"), "recipe": "recipes/vww.py",
                       "recipe_sha256": recipe_hash, "recipe_commit": recipe_commit(),
                       "recipe_files": recipe_files, "numpy": np.__version__, "onnx": onnx.__version__,
                       "tflite-micro": micro_version()},
            "evaluation": evaluation,
            "files": [{"path": p.name, "size": p.stat().st_size, "sha256": digest(p)}
                      for p in sorted(directory.iterdir())],
        }
        (directory / "artifact.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"id": metadata["id"], "stages": stages,
                          "evaluation": evaluation}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / ".build", help="Download cache for model and dataset")
    parser.add_argument("--compiler-source", type=Path, default=ROOT / ".build/compiler-vww")
    parser.add_argument("--runtime-source", type=Path, default=ROOT / ".build/runtime-vww")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, AssertionError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
