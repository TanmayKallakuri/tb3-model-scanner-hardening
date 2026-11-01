"""Keras model configuration scanning (HDF5 ``model_config`` and ``.keras`` config.json).

A ``__lambda__`` object carries marshalled bytecode that Keras materialises
into a function when it rebuilds the model: in a ``Lambda`` layer's
``function`` or ``output_shape``, or in any layer's activation (Keras
deserialises activations, initializers, regularizers and constraints through
the same path).  Named function references, named activations and every
other layer type are clean.  Layers are found wherever the JSON nests them.
"""
from __future__ import annotations

import binascii
import io
import json

CODE = "code-execution"

# layer config keys Keras deserialises with deserialize_keras_object
DESERIALIZED_KEYS = {
    "activation", "recurrent_activation", "kernel_initializer", "bias_initializer",
    "recurrent_initializer", "depthwise_initializer", "pointwise_initializer",
    "embeddings_initializer", "beta_initializer", "gamma_initializer",
    "moving_mean_initializer", "moving_variance_initializer", "kernel_regularizer",
    "bias_regularizer", "recurrent_regularizer", "activity_regularizer",
    "depthwise_regularizer", "pointwise_regularizer", "embeddings_regularizer",
    "beta_regularizer", "gamma_regularizer", "kernel_constraint", "bias_constraint",
    "recurrent_constraint", "depthwise_constraint", "pointwise_constraint",
    "embeddings_constraint", "beta_constraint", "gamma_constraint",
}


def _has_code(value):
    """True when Keras' func_load would get a marshalled code object out of ``value``."""
    if isinstance(value, (list, tuple)) and value:
        return _has_code(value[0])
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        raw = binascii.a2b_base64(value.encode("ascii"))
    except (UnicodeEncodeError, binascii.Error, ValueError):
        raw = value.encode("raw_unicode_escape")
    # marshal type code for a code object, with or without the FLAG_REF bit
    return raw[:1] in (b"c", b"\xe3")


def is_lambda_object(v):
    if not isinstance(v, dict) or v.get("class_name") != "__lambda__":
        return False
    cfg = v.get("config")
    if not isinstance(cfg, dict):
        return False
    return _has_code(cfg.get("code")) or _has_code(cfg.get("value"))


def _is_lambda_layer(class_name):
    return class_name == "Lambda" or class_name.endswith((".Lambda", ">Lambda"))


def _legacy_lambda(node):
    """A Keras 2 Lambda config: saving_utils.model_from_config turns its list-form
    function [code, defaults, closure] into a __lambda__ and materialises it."""
    conf = node.get("config")
    return (node.get("class_name") == "Lambda" and "module" not in node and isinstance(conf, dict)
            and isinstance(conf.get("function"), (list, tuple)) and _has_code(conf["function"]))


def scan_keras_json(ctx, cfg, location, legacy_top=False):
    """Walk a model config; report every place Keras would materialise bytecode."""
    if legacy_top and isinstance(cfg, dict) and _legacy_lambda(cfg):
        ctx.report(CODE, f"{location}:Lambda", "Lambda function carries serialized bytecode (legacy format)")
    # layers of (nested) models are rebuilt through model_from_config when they
    # lack a "module" key; the layer held by a wrapper is not
    for node, path in _legacy_model_layers(cfg, location):
        if _legacy_lambda(node):
            ctx.report(CODE, f"{path}:Lambda", "Lambda function carries serialized bytecode (legacy format)")
    stack = [(cfg, location, False)]
    seen = 0
    while stack:
        node, path, any_lambda = stack.pop()
        seen += 1
        if seen > 2_000_000:
            ctx.report("unsupported", location, "model configuration too large to analyse")
            return
        if isinstance(node, dict):
            cls = node.get("class_name")
            conf = node.get("config")
            if isinstance(cls, str) and isinstance(conf, dict):
                name = conf.get("name", cls)
                where = f"{path}:{cls}[{name}]"
                if _is_lambda_layer(cls):
                    for key in ("function", "output_shape"):
                        if is_lambda_object(conf.get(key)):
                            ctx.report(CODE, where, f"Lambda {key} carries serialized bytecode")
                for key in DESERIALIZED_KEYS:
                    if is_lambda_object(conf.get(key)):
                        ctx.report(CODE, where, f"{key} is a __lambda__ carrying serialized bytecode")
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    child_any = any_lambda or k in ("compile_config", "loss", "metrics", "weighted_metrics",
                                                    "loss_weights", "optimizer_config")
                    if child_any and is_lambda_object(v):
                        ctx.report(CODE, f"{path}.{k}", "training configuration holds a __lambda__ with bytecode")
                    stack.append((v, f"{path}.{k}" if not isinstance(k, str) or len(path) < 400 else path, child_any))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, (dict, list)):
                    if any_lambda and is_lambda_object(v):
                        ctx.report(CODE, f"{path}[{i}]", "training configuration holds a __lambda__ with bytecode")
                    stack.append((v, f"{path}[{i}]" if len(path) < 400 else path, any_lambda))


def _legacy_model_layers(cfg, location):
    """Layer dicts reached through models' "layers" lists (any depth)."""
    out = []
    stack = [(cfg, location)]
    seen = 0
    while stack and seen < 200000:
        node, path = stack.pop()
        seen += 1
        if not isinstance(node, dict):
            continue
        conf = node.get("config")
        layers = conf.get("layers") if isinstance(conf, dict) else None
        if isinstance(layers, list):
            for i, layer in enumerate(layers):
                if isinstance(layer, dict):
                    out.append((layer, f"{path}.layers[{i}]"))
                    stack.append((layer, f"{path}.layers[{i}]"))
    return out


def _load_json(ctx, raw, location):
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except (ValueError, RecursionError) as exc:
        ctx.note(f"{location}: configuration is not JSON ({exc})")
        return None


def looks_like_keras_config(raw):
    """config.json of a Keras v3 archive: a serialized model object."""
    try:
        cfg = json.loads(bytes(raw).decode("utf-8", "replace"))
    except (ValueError, RecursionError):
        return False
    return isinstance(cfg, dict) and isinstance(cfg.get("class_name"), str) and isinstance(cfg.get("config"), dict)


def scan_keras_config_bytes(ctx, raw, location):
    cfg = _load_json(ctx, raw, location)
    if cfg is not None:
        scan_keras_json(ctx, cfg, location)


def _attr_text(value):
    if not isinstance(value, (bytes, str)):
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                value = value.reshape(-1)[0] if value.size else b""
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def scan_h5(ctx, data, location):
    """Keras HDF5: the root attribute model_config (and training_config) hold JSON."""
    from modelgate import h5lite
    try:
        attrs = h5lite.root_attributes(data)
    except h5lite.H5Corrupt as exc:
        ctx.report("unsupported", location, f"HDF5 file is malformed: {exc}")
        return
    except h5lite.H5Unsupported:
        attrs = None  # a layout only the HDF5 library reads
    if attrs is not None:
        for attr in ("model_config", "training_config"):
            if attr in attrs:
                _scan_attr(ctx, attr, _attr_text(attrs[attr]), location)
        return
    _scan_h5_library(ctx, data, location)


def _scan_attr(ctx, attr, raw, location):
    cfg = _load_json(ctx, raw, f"{location}:{attr}")
    if cfg is None:
        return
    if attr == "model_config":
        scan_keras_json(ctx, cfg, f"{location}:{attr}", legacy_top=True)
    else:
        scan_keras_json(ctx, {"compile_config": cfg}, f"{location}:{attr}")


def _scan_h5_library(ctx, data, location):
    try:
        import h5py
        fh = h5py.File(io.BytesIO(data), "r")
    except Exception as exc:  # noqa: BLE001 - h5py raises many types
        ctx.report("unsupported", location, f"HDF5 file cannot be opened: {exc}")
        return
    with fh:
        for attr in ("model_config", "training_config"):
            try:
                if attr not in fh.attrs:
                    continue
                raw = _attr_text(fh.attrs[attr])
            except Exception as exc:  # noqa: BLE001
                ctx.report("unsupported", f"{location}:{attr}", f"attribute cannot be read: {exc}")
                continue
            _scan_attr(ctx, attr, raw, location)
