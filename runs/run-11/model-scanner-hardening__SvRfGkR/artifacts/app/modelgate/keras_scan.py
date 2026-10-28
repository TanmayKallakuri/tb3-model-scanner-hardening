"""Inspect Keras configuration as data, without deserializing any objects."""
import io
import json

from modelgate.report import Finding
from modelgate.limits import MAX_ITEMS, LimitExceeded


def has_bytecode(value):
    if not isinstance(value, dict) or value.get("class_name") != "__lambda__":
        return False
    config = value.get("config")
    return isinstance(config, dict) and ("code" in config or "value" in config)


def scan_config(config, location):
    if isinstance(config, (str, bytes)):
        config = json.loads(config)
    findings = []
    todo = [(config, location)]
    count = 0
    while todo:
        value, path = todo.pop()
        count += 1
        if count > MAX_ITEMS:
            raise LimitExceeded("Keras configuration is too large")
        if isinstance(value, dict):
            cfg = value.get("config")
            if isinstance(cfg, dict) and isinstance(value.get("class_name"), str):
                fields = ["activation"]
                if value["class_name"] == "Lambda":
                    fields += ["function", "output_shape"]
                for field in fields:
                    fn = cfg.get(field)
                    legacy = (value["class_name"] == "Lambda"
                              and field in ("function", "output_shape")
                              and cfg.get(field + "_type") == "lambda"
                              and isinstance(fn, (list, tuple)) and len(fn) == 3)
                    if has_bytecode(fn) or legacy:
                        findings.append(Finding("code-execution", f"{path}.config.{field}",
                                                "Keras deserializes Python bytecode"))
                        if len(findings) >= 100:
                            return findings
            todo.extend((v, f"{path}.{k}") for k, v in value.items() if isinstance(v, (dict, list)))
        elif isinstance(value, list):
            todo.extend((v, f"{path}[{i}]") for i, v in enumerate(value) if isinstance(v, (dict, list)))
    return findings


def scan_h5(source, location):
    # h5py's version initialization indexes platform.uname(). On Python 3.12
    # that also evaluates its lazy `processor` field, which otherwise runs
    # `uname -p` in a subprocess. Populate the cache from the uname syscall so
    # even importing our trusted HDF5 parser remains entirely process-free.
    import platform
    uname = platform.uname()
    uname.__dict__.setdefault('processor', uname.machine)
    import h5py
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    # Never extract a container member or open links/datasets in the HDF5.
    with h5py.File(source, "r") as fh:
        raw = fh.attrs.get("model_config")
    return [] if raw is None else scan_config(raw, location + ":model_config")
