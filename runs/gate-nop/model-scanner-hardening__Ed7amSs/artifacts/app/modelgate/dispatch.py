"""Choose a scanner by artifact type and build the result."""
from __future__ import annotations

import io
import os
import zipfile

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.report import Result

PICKLE_EXTS = (".pkl", ".pickle", ".joblib")
TORCH_EXTS = (".pt", ".pth", ".ckpt", ".bin")  # .bin: a torch checkpoint or a plain pickle
H5_EXTS = (".h5", ".hdf5")


def scan_path(path: str) -> Result:
    name = os.path.basename(path.rstrip("/"))
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    ext = os.path.splitext(name)[1].lower()
    if ext in PICKLE_EXTS:
        with open(path, "rb") as fh:
            return Result(path, pickle_scan.scan_pickle(fh.read(), name))
    if ext in TORCH_EXTS:
        return Result(path, _scan_torch(path, name))
    if ext in H5_EXTS:
        return Result(path, keras_scan.scan_h5(path, name))
    # .npy, .npz, .keras and compressed archives are not model formats the
    # scanner understands; nothing to report for them
    return Result(path, [])


def _scan_torch(path: str, name: str):
    """A torch checkpoint is a zip whose data.pkl holds the object graph."""
    with open(path, "rb") as fh:
        data = fh.read()
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return pickle_scan.scan_pickle(data, name)  # legacy (pre-1.6) checkpoint: a pickle
    findings = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if member.endswith("/data.pkl") or member == "data.pkl":
                findings.extend(pickle_scan.scan_pickle(zf.read(member), f"{name}!{member}"))
    return findings
