"""Inspect Keras configuration data; never deserialize layers or functions."""
from __future__ import annotations
import json
from modelgate.report import Finding


def _bytecode(value):
    if not isinstance(value, dict) or value.get('class_name') != '__lambda__':
        return False
    config = value.get('config', {})
    return isinstance(config, dict) and (bool(config.get('code')) or bool(config.get('value')))


def scan_config(config, location: str) -> list[Finding]:
    findings = []
    todo = [(config, location)]
    while todo:
        value, where = todo.pop()
        if isinstance(value, list):
            todo.extend((v, f'{where}[{i}]') for i, v in enumerate(value))
        elif isinstance(value, dict):
            cfg = value.get('config')
            if isinstance(cfg, dict) and isinstance(value.get('class_name'), str):
                if value['class_name'] == 'Lambda':
                    for field in ('function', 'output_shape'):
                        if _bytecode(cfg.get(field)):
                            findings.append(Finding('code-execution', f'{where}.config.{field}',
                                                    'Keras serialized lambda bytecode'))
                        elif (cfg.get(field + '_type') == 'lambda' and
                              isinstance(cfg.get(field), (list, tuple)) and cfg[field]):
                            findings.append(Finding('code-execution', f'{where}.config.{field}',
                                                    'Keras legacy lambda bytecode'))
                if _bytecode(cfg.get('activation')):
                    findings.append(Finding('code-execution', f'{where}.config.activation',
                                            'Keras serialized activation bytecode'))
            todo.extend((v, f'{where}.{k}') for k, v in value.items()
                        if isinstance(v, (dict, list)))
    return findings


def scan_json(raw, location: str) -> list[Finding]:
    return scan_config(json.loads(raw), location)


def scan_h5(path, location: str) -> list[Finding]:
    # NumPy/h5py asks for platform.uname() as a tuple. Its lazy processor
    # field otherwise invokes `uname -p` in a subprocess on this platform.
    # Cache the already available kernel machine identifier instead.
    import platform
    uname = platform.uname()
    uname.__dict__['processor'] = uname.machine
    import h5py
    # BytesIO works as well as a path. Do not traverse datasets or links.
    with h5py.File(path, 'r') as fh:
        raw = fh.attrs.get('model_config')
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return scan_json(raw, f'{location}:model_config')
