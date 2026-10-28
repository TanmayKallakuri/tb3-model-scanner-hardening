#!/usr/bin/env python3
"""Capture real, loadable TensorFlow SavedModel and Keras skeletons as fixtures.

Runs INSIDE the throwaway `gtval` image (tensorflow-cpu 2.19.0, keras 3.10.0):

    MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/repo" gtval:latest \
        python /repo/scripts/gt-validate/make_fixtures.py

The corpus generator (tests/gen/savedmodel.py, tests/gen/keras_files.py)
never imports TensorFlow or Keras: it mutates these fixtures at the protobuf
and JSON level.  Every fixture is tiny and benign.  SavedModels are stored as
tars with zeroed metadata so the fixture bytes are deterministic.
"""
import hashlib
import io
import json
import os
import pathlib
import shutil
import tarfile
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

OUT = pathlib.Path("/repo/tasks/model-scanner-hardening/tests/gen/fixtures")


def tar_dir(root: pathlib.Path) -> bytes:
    """Deterministic tar of a directory: sorted members, zero times and ids."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(p for p in root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            info = tarfile.TarInfo(rel)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
            else:
                data = path.read_bytes()
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ----------------------------------------------------------------------------
# TensorFlow skeletons
# ----------------------------------------------------------------------------


class MatMulModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(np.full((4, 2), 0.5, dtype=np.float32), name="w")
        self.b = tf.Variable(np.zeros((2,), dtype=np.float32), name="b")

    @tf.function(input_signature=[tf.TensorSpec((None, 4), tf.float32, name="x")])
    def serve(self, x):
        return tf.identity(tf.nn.bias_add(tf.matmul(x, self.w), self.b), name="output")


class CondModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(np.full((4, 2), 0.5, dtype=np.float32), name="w")

    @tf.function(input_signature=[tf.TensorSpec((None, 4), tf.float32, name="x")])
    def serve(self, x):
        y = tf.matmul(x, self.w)
        total = tf.reduce_sum(y)
        scale = tf.cond(total >= 0.0, lambda: tf.constant(1.0), lambda: tf.constant(0.5))
        return tf.identity(y * scale, name="output")


class LoopModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(np.full((4, 4), 0.1, dtype=np.float32), name="w")

    @tf.function(input_signature=[tf.TensorSpec((None, 4), tf.float32, name="x")])
    def serve(self, x):
        i = tf.constant(0)
        y = x

        def cond(i, y):
            return i < 3

        def body(i, y):
            return i + 1, tf.matmul(y, self.w)

        _, y = tf.while_loop(cond, body, [i, y], maximum_iterations=3)
        return tf.identity(y, name="output")


class TwoSignatureModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(np.full((4, 2), 0.5, dtype=np.float32), name="w")

    @tf.function(input_signature=[tf.TensorSpec((None, 4), tf.float32, name="x")])
    def serve(self, x):
        return tf.identity(tf.matmul(x, self.w), name="output")

    @tf.function(input_signature=[tf.TensorSpec((None,), tf.string, name="text")])
    def describe(self, text):
        return tf.identity(tf.strings.length(text), name="length")


def save_tf(name, module, signatures):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        tf.saved_model.save(module, path, signatures=signatures)
        fp = pathlib.Path(path, "fingerprint.pb")
        if fp.exists():
            fp.unlink()
        # prove it loads and runs before capturing
        loaded = tf.saved_model.load(path)
        for sig_name, fn in loaded.signatures.items():
            kwargs = {}
            for k, spec in fn.structured_input_signature[1].items():
                shape = tuple(1 if d is None else d for d in spec.shape)
                kwargs[k] = tf.constant(np.full(shape, "x")) if spec.dtype == tf.string else tf.zeros(shape, spec.dtype)
            fn(**kwargs)
        data = tar_dir(pathlib.Path(path))
    out = OUT / "tf" / f"{name}.tar"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out, data


# ----------------------------------------------------------------------------
# Keras skeletons
# ----------------------------------------------------------------------------


def keras_models():
    inp = keras.Input((4,), name="features")
    x = keras.layers.Dense(3, activation="relu", name="dense")(inp)
    x = keras.layers.Dense(2, name="dense_1")(x)
    out = keras.layers.Dense(1, name="score")(x)
    functional = keras.Model(inp, out, name="functional_dense")

    inner = keras.Sequential([keras.Input((3,)), keras.layers.Dense(2, name="inner_dense")], name="inner_block")
    nested = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3, name="dense"), inner, keras.layers.Dense(1, name="head")], name="nested_sequential")

    inp2 = keras.Input((5, 4), name="sequence")
    y = keras.layers.TimeDistributed(keras.layers.Dense(3, name="td_dense"), name="time_distributed")(inp2)
    y = keras.layers.Bidirectional(keras.layers.LSTM(2, name="lstm"), name="bidirectional")(y)
    out2 = keras.layers.Dense(1, name="score")(y)
    wrappers = keras.Model(inp2, out2, name="wrapper_model")

    inp3 = keras.Input((8, 2), name="signal")
    z = keras.layers.Conv1D(2, 3, name="conv1d")(inp3)
    z = keras.layers.GlobalAveragePooling1D(name="pool")(z)
    out3 = keras.layers.Dense(1, name="score")(z)
    conv = keras.Model(inp3, out3, name="conv_model")
    return {"functional": functional, "nested": nested, "wrappers": wrappers, "conv": conv}


def save_keras(name, model):
    with tempfile.TemporaryDirectory() as tmp:
        outs = {}
        for ext in (".keras", ".h5"):
            path = os.path.join(tmp, name + ext)
            model.save(path)
            reloaded = keras.saving.load_model(path)
            shape = tuple(1 if d is None else d for d in reloaded.inputs[0].shape)
            reloaded(np.zeros(shape, dtype="float32"))
            data = pathlib.Path(path).read_bytes()
            out = OUT / "keras" / (name + ext)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            outs[ext] = (out, data)
    return outs


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = {"tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__, "fixtures": {}}
    for name, module, sigs in [
        ("matmul", MatMulModule(), None),
        ("cond", CondModule(), None),
        ("loop", LoopModule(), None),
        ("two_signatures", TwoSignatureModule(), None),
    ]:
        sigs = {"serving_default": module.serve} if sigs is None else sigs
        if name == "two_signatures":
            sigs = {"serving_default": module.serve, "describe": module.describe}
        out, data = save_tf(name, module, sigs)
        manifest["fixtures"][out.relative_to(OUT).as_posix()] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        print("tf", name, len(data), "bytes")
    for name, model in keras_models().items():
        for ext, (out, data) in save_keras(name, model).items():
            manifest["fixtures"][out.relative_to(OUT).as_posix()] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            print("keras", name + ext, len(data), "bytes")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
