"""Read Keras configuration as data; never import Keras or deserialize a layer."""
from __future__ import annotations
import json
from modelgate.report import Finding


def _bytecode(value):
    if not isinstance(value, dict) or value.get('class_name') != '__lambda__':
        return False
    config = value.get('config', {})
    return isinstance(config, dict) and ('code' in config or 'value' in config)


def scan_config(config, location):
    findings = []
    pending = [(config, location)]
    while pending:
        value, path = pending.pop()
        if isinstance(value, dict):
            cfg = value.get('config')
            if isinstance(cfg, dict):
                if value.get('class_name') == 'Lambda':
                    for field in ('function', 'output_shape'):
                        fn = cfg.get(field)
                        # Keras 2 HDF5 stores func_dump's triple and a type tag.
                        legacy = isinstance(fn, (list, tuple)) and cfg.get(field + '_type') == 'lambda'
                        if _bytecode(fn) or legacy:
                            findings.append(Finding('code-execution', path + '.config.' + field,
                                                    'serialized Python bytecode in Lambda ' + field))
                if _bytecode(cfg.get('activation')):
                    findings.append(Finding('code-execution', path + '.config.activation',
                                            'serialized Python bytecode in activation'))
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    pending.append((child, path + '.' + str(key)))
        elif isinstance(value, list):
            pending.extend((child, f'{path}[{i}]') for i, child in enumerate(value))
    return findings


def scan_json(data, location):
    return scan_config(json.loads(data), location)


def scan_h5(path, location):
    # h5py.version formats platform.uname(), including its lazy processor
    # field. On CPython/Linux that field otherwise launches `uname -p`.
    # Populate the informational field from os.uname's machine value before
    # importing h5py. Neither probing a model nor importing its parser should
    # start a helper process.
    import platform
    uname = platform.uname()
    uname.__dict__.setdefault('processor', uname.machine)
    import h5py
    # path may be a BytesIO. h5py does no Python object deserialization here.
    with h5py.File(path, 'r') as fh:
        raw = fh.attrs.get('model_config')
    return [] if raw is None else scan_json(raw, location + ':model_config')
