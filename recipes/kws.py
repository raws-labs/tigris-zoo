"""Compile and validate the MLPerf Tiny keyword-spotting DS-CNN as an int8 plan."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import wave

import numpy as np
import onnx
import tflite
from tflite_micro.python.tflite_micro import runtime as micro

from tigris.emitters.binary.reader import read_binary_plan
from tigris.zoo import digest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from int8_common import ROOT, checkout, fetch, recipe_commit, run, runner  # noqa: E402
from tflite_qdq import convert  # noqa: E402

COMPILER = "908b0fd0a8687f45e20594ba95977aa8695c26ff"
COMPILER_VERSION = "0.11.3"
RUNTIME = "96ed57e1d9122db779f9e8d388a81098e0e8f27c"
RUNTIME_VERSION = "0.11.3"
MODEL = ("https://raw.githubusercontent.com/mlcommons/tiny/4addd0fa08d216e20637637874e084895f289da4/"
         "benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite")
MODEL_SHA = "aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
DATASET = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
DATASET_SHA = "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df"
LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "_silence_", "_unknown_"]
# MLPerf Tiny's top-1 quality target for this benchmark.
ACCURACY_FLOOR = 0.90
BUDGET = 16 * 1024
INPUT_SHAPE = [1, 49, 10, 1]


def _mel_matrix(bins=257, mels=40, rate=16000, low=20.0, high=4000.0):
    """tf.signal.linear_to_mel_weight_matrix."""
    def hz_to_mel(f):
        return 1127.0 * np.log1p(f / 700.0)
    spectrum_mel = hz_to_mel(np.linspace(0.0, rate / 2, bins)[1:])[:, None]
    edges = np.linspace(hz_to_mel(low), hz_to_mel(high), mels + 2)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    weights = np.maximum(0.0, np.minimum((spectrum_mel - lower) / (center - lower),
                                         (upper - spectrum_mel) / (upper - center)))
    return np.vstack([np.zeros((1, mels)), weights])


MEL = _mel_matrix()
WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(480) / 480)
DCT = np.array([[2 * np.cos(np.pi * k * (2 * n + 1) / 80) for n in range(40)] for k in range(10)]) / np.sqrt(80)


def mfcc(samples):
    """MLPerf Tiny's KWS features: peak-normalized 1 s of 16 kHz audio, 30 ms
    Hann windows every 20 ms, 40 log-mel bins from 20 Hz to 4 kHz, 10 DCT-II
    coefficients. Returns float32 [49, 10]."""
    x = samples.astype(np.float64)
    if x.max() <= 0:
        raise ValueError("clip has no positive sample to normalize by")
    x = np.pad(x / x.max(), (0, 16000))[:16000]
    frames = np.stack([x[i * 320:i * 320 + 480] * WINDOW for i in range(49)])
    log_mel = np.log(np.abs(np.fft.rfft(frames, n=512)) @ MEL + 1e-6)
    return (log_mel @ DCT.T).astype(np.float32)


def clips(archive):
    """Every test clip in name order: features, labels, names."""
    audio = wavs(archive)
    features, labels, names = [], [], sorted(audio)
    for name in names:
        with wave.open(io.BytesIO(audio[name])) as w:
            if (w.getframerate(), w.getsampwidth(), w.getnchannels()) != (16000, 2, 1):
                raise ValueError(f"unexpected audio format: {name}")
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        features.append(mfcc(samples))
        labels.append(LABELS.index(name.split("/")[-2]))
    return np.array(features), np.array(labels), names


def wavs(archive):
    """Every clip's bytes by name, read in one pass over the compressed archive."""
    with tarfile.open(archive, "r:gz") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar if m.isfile() and m.name.endswith(".wav")}


def input_quantization(model_bytes):
    graph = tflite.Model.GetRootAs(model_bytes, 0).Subgraphs(0)
    q = graph.Tensors(graph.Inputs(0)).Quantization()
    return float(q.ScaleAsNumpy()[0]), int(q.ZeroPointAsNumpy()[0])


def quantize(features, scale, zero_point):
    """The runtime's float32 input conversion: divide in float32, round half
    away from zero, saturate."""
    scaled = features / np.float32(scale)
    return np.clip(np.trunc(scaled + np.copysign(np.float32(0.5), scaled)) + zero_point,
                   -128, 127).astype(np.int8)


def references(model_path, quantized):
    """TFLite Micro's int8 scores for every clip."""
    interpreter = micro.Interpreter.from_file(str(model_path), arena_size=64 * 1024)
    outputs = []
    for row in quantized:
        interpreter.set_input(row.reshape(INPUT_SHAPE), 0)
        interpreter.invoke()
        outputs.append(interpreter.get_output(0).reshape(-1).copy())
    return np.array(outputs, dtype=np.int8)


def model(output, data):
    """The source model, its ONNX conversion, and the quantized test set."""
    tflite_path = fetch(MODEL, data / "kws_ref_model.tflite", MODEL_SHA)
    archive = fetch(DATASET, data / "speech_commands_test_set_v0.02.tar.gz", DATASET_SHA)
    model_bytes = tflite_path.read_bytes()
    onnx.save(convert(model_bytes, "kws_ds_cnn"), output / "model.onnx")
    features, labels, names = clips(archive)
    scale, zero_point = input_quantization(model_bytes)
    quantized = quantize(features, scale, zero_point)
    reference = references(tflite_path, quantized)
    metrics = {
        "test_clips": len(labels), "labels": LABELS,
        "reference": "TFLite Micro reference kernels", "reference_package": f"tflite-micro {micro_version()}",
        "reference_accuracy": float((reference.astype(np.int32).argmax(axis=1) == labels).mean()),
        "accuracy_floor": ACCURACY_FLOOR,
        "tested_runtime_versions": [RUNTIME_VERSION], "runtime_commit": RUNTIME,
        "scope": "Speech Commands v0.02 test set as distributed; no other speakers or microphones.",
    }
    return features, labels, names, quantized, reference, metrics, archive


def micro_version():
    from importlib.metadata import version
    return version("tflite-micro")


def runtime_gates(executable, plan, quantized, reference, labels, metrics, fast_bytes, workdir):
    """The acceptance gates for a plan on one runtime: int8 scores identical to
    TFLite Micro on every clip, accuracy at or above the floor, the fast budget,
    and refusal of a slow arena one byte short. Raises on any failure."""
    quantized.tofile(workdir / "inputs.bin")
    report = json.loads(run([executable, plan, workdir / "inputs.bin", workdir / "outputs.bin"]))
    scores = np.fromfile(workdir / "outputs.bin", dtype=np.int8).reshape(reference.shape)
    if report["samples"] != len(reference):
        raise ValueError("runner did not process every clip")
    differing = int((scores != reference).any(axis=1).sum())
    accuracy = float((scores.astype(np.int32).argmax(axis=1) == labels).mean())
    evaluation = dict(metrics, parity_passed=differing == 0, clips_differing=differing,
                      max_abs_lsb=int(np.abs(scores.astype(np.int32) - reference).max()),
                      runtime_accuracy=accuracy,
                      measured_fast_peak_bytes=report["fast_peak"],
                      measured_slow_bytes=report["slow_peak"],
                      host_executor_workspace_bytes=report["workspace_bytes"])
    if differing:
        raise ValueError(f"runtime scores differ from TFLite Micro on {differing} clips")
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
    _, labels, _, quantized, reference, metrics, _ = model(output, data)
    if digest(output / "model.onnx") != source["onnx_sha256"]:
        raise ValueError("conversion did not reproduce the published model")
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
    data = args.data.resolve()
    features, labels, names, quantized, reference, metrics, archive = model(output, data)
    if metrics["reference_accuracy"] < ACCURACY_FLOOR:
        raise ValueError("the source model misses the accuracy floor on this test set")
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
    evaluation = runtime_gates(executable, directory / "model.tgrs", quantized, reference, labels, metrics,
                               BUDGET, output)

    # The first clip in name order, chosen without consulting any prediction.
    features[0].reshape(INPUT_SHAPE).astype("<f4").tofile(directory / "example-input.bin")
    ((reference[0].astype(np.float32) + 128) / 256).astype("<f4").tofile(directory / "example-output.bin")
    (directory / "example.wav").write_bytes(wavs(archive)[names[0]])
    (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    (directory / "license.txt").write_text(
        (ROOT / "LICENSE").read_text() + "\nModel attribution:\n"
        "MLCommons MLPerf Tiny keyword-spotting reference model (kws_ref_model.tflite),\n"
        "https://github.com/mlcommons/tiny, Apache License 2.0.\n"
        "\nDataset and example data attribution:\n"
        "Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech\n"
        "Recognition. https://arxiv.org/abs/1804.03209\n"
        "Speech Commands v0.02 and the derived example audio and features: CC BY 4.0,\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        "example.wav is an unmodified test clip; the example tensors are its MFCC\n"
        "features and the reference scores.\n"
    )
    (directory / "readme.md").write_text(
        "# Keyword spotting (DS-CNN, int8)\n\n"
        "The MLPerf Tiny keyword-spotting reference model, a depthwise-separable CNN,\n"
        "classifies one second of 16 kHz mono audio into 12 classes:\n"
        f"{', '.join(LABELS)}.\n\n"
        "Input: float32 [1, 49, 10, 1] MFCC features, quantized by the runtime at the\n"
        "boundary. Peak-normalize the clip by its largest sample, zero-pad or cut it\n"
        "to 16000 samples, take 49 periodic Hann windows of 480 samples every 320\n"
        "samples, a 512-point FFT magnitude, 40 mel bins from 20 Hz to 4 kHz\n"
        "(tf.signal.linear_to_mel_weight_matrix), log(x + 1e-6), and the first 10\n"
        "DCT-II coefficients scaled by 1/sqrt(80) (tf.signal.mfccs_from_log_mel_spectrograms).\n"
        "Output: float32 [1, 12] class probabilities from an int8 Softmax, in\n"
        "steps of 1/256.\n\n"
        f"On the {metrics['test_clips']} clips of the Speech Commands v0.02 test set the\n"
        f"runtime's int8 scores equal TFLite Micro's on every clip, and top-1 accuracy is\n"
        f"{evaluation['runtime_accuracy']:.4f}. The recipe requires {ACCURACY_FLOOR:.2f}, MLPerf Tiny's\n"
        "quality target for this benchmark.\n\n"
        "example.wav is the first test clip in name order. example-input.bin holds its\n"
        "features and example-output.bin the TFLite Micro scores dequantized to\n"
        "float32; `tigris run` reproduces them exactly.\n\n"
        f"The plan uses int8 reference kernels with weights read in place from flash.\n"
        f"The {BUDGET // 1024} KiB fast arena holds all activations; the "
        f"{evaluation['measured_slow_bytes']}-byte slow arena holds\n"
        "the model input and output. Smaller fast budgets tile the model but move\n"
        "16000 bytes of activations into the slow arena, more RAM in total, so only\n"
        "this variant is built. Arena requirements exclude stack, runtime\n"
        "structures, and executor workspace. Tested on the host with runtime\n"
        f"{RUNTIME_VERSION}; no hardware latency measurements are claimed. Runtime 0.11.1\n"
        "or newer is required: older runtimes refuse the plan's average-pool rounding\n"
        "attribute. Current constraints and validation history are preserved in\n"
        "download.json when this artifact is downloaded.\n"
    )
    build_hash = hashlib.sha256((recipe_hash + "".join(recipe_files.values()) + digest(directory / "model.tgrs")
                                 + COMPILER + RUNTIME).encode()).hexdigest()[:16]
    metadata = {
        "format_version": 1, "id": f"kws-ds-cnn-s8-s9-{BUDGET // 1024}k-{build_hash}",
        "model": "kws-ds-cnn", "category": "keyword-spotting", "schema": 9,
        "quantization": "int8", "backends": ["reference"],
        "runtime": {"min": "0.11.1", "max": None},
        "compatibility_note": ("Schema 9 int8 plan with the average-pool rounding attribute, which runtimes "
                               "before 0.11.1 refuse; no known upper cutoff."),
        "memory": {"fast_bytes": BUDGET, "slow_bytes": evaluation["measured_slow_bytes"],
                   "flash_bytes": (directory / "model.tgrs").stat().st_size},
        "inputs": [{"name": "input", "shape": INPUT_SHAPE, "dtype": "float32"}],
        "outputs": [{"name": "output", "shape": [1, 12], "dtype": "float32"}],
        "license": "apache-2.0 AND cc-by-4.0",
        "compiler": {"version": COMPILER_VERSION, "commit": COMPILER},
        "source": {"model": MODEL, "model_sha256": MODEL_SHA, "dataset": DATASET, "dataset_sha256": DATASET_SHA,
                   "onnx_sha256": digest(output / "model.onnx"), "recipe": "recipes/kws.py",
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
    parser.add_argument("--compiler-source", type=Path, default=ROOT / ".build/compiler-kws")
    parser.add_argument("--runtime-source", type=Path, default=ROOT / ".build/runtime-kws")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, AssertionError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
