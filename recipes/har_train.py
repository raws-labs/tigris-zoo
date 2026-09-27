"""Train and quantize the activity-recognition 1D CNN once; the recipe pins the result.

Training is not bit-reproducible across CPUs, so its int8 QDQ ONNX output is
published as a source file and recipes/har.py fetches it by SHA-256. Needs
PyTorch in addition to the build dependencies:

    python recipes/har_train.py --dataset .build/har.zip --output har_int8.onnx
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from har_data import LABELS, load, normalization, normalize  # noqa: E402

VALIDATION_SUBJECTS = 5
EPOCHS = 40
SEED = 0
WIDTH = 32
DROPOUT = 0.3
WEIGHT_DECAY = 1e-3


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, WIDTH, 9, stride=2, padding=4), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Conv1d(WIDTH, 2 * WIDTH, 5, stride=2, padding=2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Conv1d(2 * WIDTH, 2 * WIDTH, 5, padding=2), nn.ReLU(),
        )
        self.classifier = nn.Linear(2 * WIDTH, len(LABELS))

    def forward(self, x):
        return self.classifier(self.features(x).mean(dim=2))


def accuracy(model, windows, labels):
    model.eval()
    with torch.no_grad():
        return float((model(torch.from_numpy(windows)).argmax(dim=1).numpy() == labels).mean())


class Calibration(CalibrationDataReader):
    def __init__(self, windows):
        self.rows = iter(windows[:, None])

    def get_next(self):
        row = next(self.rows, None)
        return None if row is None else {"input": row}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    splits = load(args.dataset)
    windows, labels, subjects = splits["train"]
    mean, std = normalization(windows)
    windows = normalize(windows, mean, std)
    held_out = np.isin(subjects, np.unique(subjects)[-VALIDATION_SUBJECTS:])
    fit_x, fit_y = torch.from_numpy(windows[~held_out]), torch.from_numpy(labels[~held_out])

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    best = (-1.0, 0, None)
    for epoch in range(EPOCHS):
        model.train()
        for batch in torch.randperm(len(fit_x), generator=generator).split(64):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(fit_x[batch]), fit_y[batch]).backward()
            optimizer.step()
        score = accuracy(model, windows[held_out], labels[held_out])
        if score > best[0]:
            best = (score, epoch, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    model.eval()

    float_path = args.output.with_suffix(".float.onnx")
    torch.onnx.export(model, torch.zeros(1, 6, 128), float_path, input_names=["input"],
                      output_names=["output"], opset_version=17, dynamo=False)
    calibration = windows[~held_out][::16]
    quantize_static(str(float_path), str(args.output), Calibration(calibration),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    quantized = onnx.load(args.output)
    onnx.helper.set_model_props(quantized, {
        "labels": json.dumps(LABELS),
        "normalization_mean": json.dumps(mean.tolist()),
        "normalization_std": json.dumps(std.tolist()),
    })
    onnx.save(quantized, args.output)
    test_x, test_y, _ = splits["test"]
    print(json.dumps({
        "validation_accuracy": best[0], "epoch": best[1], "calibration_windows": len(calibration),
        "float_test_accuracy": accuracy(model, normalize(test_x, mean, std), test_y),
        "torch": torch.__version__,
    }))


if __name__ == "__main__":
    main()
