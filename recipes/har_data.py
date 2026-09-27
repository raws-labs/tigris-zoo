"""UCI HAR windows for the activity-recognition recipe and its training script."""

import io
import zipfile

import numpy as np

DATASET = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATASET_SHA = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
LABELS = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]
# Accelerometer in g, gyroscope in rad/s, 128 samples at 50 Hz per window.
CHANNELS = ["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]


def load(archive):
    """{"train"|"test": (windows float32 [N, 6, 128], labels [N] in 0..5, subjects [N])}."""
    with zipfile.ZipFile(archive) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip")))

    def table(name, dtype):
        return np.loadtxt(io.StringIO(inner.read(f"UCI HAR Dataset/{name}").decode()), dtype=dtype, ndmin=2)

    splits = {}
    for split in ("train", "test"):
        signals = [table(f"{split}/Inertial Signals/{channel}_{split}.txt", np.float64) for channel in CHANNELS]
        windows = np.stack(signals, axis=1).astype(np.float32)
        labels = table(f"{split}/y_{split}.txt", np.int64)[:, 0] - 1
        subjects = table(f"{split}/subject_{split}.txt", np.int64)[:, 0]
        if windows.shape[1:] != (6, 128) or not (len(windows) == len(labels) == len(subjects)):
            raise ValueError(f"unexpected {split} split layout")
        splits[split] = windows, labels, subjects
    return splits


def normalization(train_windows):
    """Per-channel mean and population standard deviation of the training windows."""
    values = train_windows.transpose(1, 0, 2).reshape(6, -1).astype(np.float64)
    return values.mean(axis=1).astype(np.float32), values.std(axis=1).astype(np.float32)


def normalize(windows, mean, std):
    return ((windows - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def baseline_accuracy(splits):
    """Test accuracy of a ridge classifier on each channel's mean and standard
    deviation, fit on every training window with penalty 1: the simple model the
    CNN has to beat."""
    def features(windows, center=None, scale=None):
        stats = np.c_[windows.mean(axis=2), windows.std(axis=2)].astype(np.float64)
        if center is None:
            center, scale = stats.mean(axis=0), stats.std(axis=0)
        return np.c_[(stats - center) / scale, np.ones(len(stats))], center, scale

    train, train_labels, _ = splits["train"]
    test, test_labels, _ = splits["test"]
    a, center, scale = features(train)
    b, _, _ = features(test, center, scale)
    penalty = np.eye(a.shape[1])
    penalty[-1, -1] = 0
    weights = np.linalg.solve(a.T @ a + penalty, a.T @ np.eye(len(LABELS))[train_labels])
    return float(((b @ weights).argmax(axis=1) == test_labels).mean())
