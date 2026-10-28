"""Choose a scanner by artifact content and build the result.

Files are classified by content, not extension (threat model section 2): the
bytes are handed to ``formats.scan_blob``, which detects pickle, numpy, HDF5,
zip/torch/keras archives and gzip/bz2/xz/tar wrappers and recurses.  A directory
is a TensorFlow SavedModel.
"""
from __future__ import annotations

import os

from modelgate import formats, savedmodel_scan
from modelgate.report import Finding, Result

_MAX_BYTES = 2 * 1024 * 1024 * 1024


def scan_path(path: str) -> Result:
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    base = os.path.basename(path.rstrip("/")) or path
    try:
        size = os.path.getsize(path)
        if size > _MAX_BYTES:
            return Result(path, [Finding("unsupported", path, "artifact too large")])
        with open(path, "rb") as fh:
            head = fh.read(8)
            # A top-level HDF5 model can be large; read it lazily from disk via
            # h5py rather than loading the whole file into memory.
            if head == b"\x89HDF\r\n\x1a\n":
                from modelgate import keras_scan
                return Result(path, keras_scan.scan_h5(path, base))
            data = head + fh.read()
    except OSError as exc:
        return Result(path, [Finding("unsupported", path, f"cannot read artifact: {exc}")])

    findings = formats.scan_blob(data, base, depth=0, top=True)
    return Result(path, findings)
