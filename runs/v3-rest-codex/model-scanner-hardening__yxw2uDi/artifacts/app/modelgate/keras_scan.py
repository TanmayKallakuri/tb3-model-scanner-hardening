"""Inspect Keras configuration data without deserializing a model."""
from __future__ import annotations

import io
import json

from modelgate.report import Finding


def _bytecode(value):
    return (isinstance(value, dict) and value.get('class_name') == '__lambda__'
            and isinstance(value.get('config'), dict)
            and ('code' in value['config'] or 'value' in value['config']))


def scan_config(config, location: str) -> list[Finding]:
    if isinstance(config, (bytes, str)):
        config = json.loads(config)
    findings = []
    todo = [(config, location)]
    while todo:
        value, path = todo.pop()
        if isinstance(value, dict):
            cfg = value.get('config')
            if isinstance(cfg, dict):
                if value.get('class_name') == 'Lambda':
                    for field in ('function', 'output_shape'):
                        if _bytecode(cfg.get(field)):
                            findings.append(Finding('code-execution', f'{path}.config.{field}',
                                                    'Keras Lambda contains serialized bytecode'))
                        # Legacy Keras stores [code, defaults, closure] with
                        # an explicit function_type/output_shape_type marker.
                        elif (isinstance(cfg.get(field), (list, tuple)) and cfg.get(field)
                              and cfg.get(field + '_type') == 'lambda'):
                            findings.append(Finding('code-execution', f'{path}.config.{field}',
                                                    'legacy Keras Lambda bytecode'))
                if _bytecode(cfg.get('activation')):
                    findings.append(Finding('code-execution', f'{path}.config.activation',
                                            'serialized activation bytecode'))
            # Descend through models, wrappers, lists, and arbitrary depths.
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    todo.append((child, f'{path}.{key}'))
        elif isinstance(value, list):
            todo.extend((child, f'{path}[{i}]') for i, child in enumerate(value)
                        if isinstance(child, (dict, list)))
    return findings


def scan_h5(source, location: str) -> list[Finding]:
    # h5py's extension queries platform.uname by tuple slicing. In Python
    # 3.12 that otherwise resolves processor by spawning `uname -p`. Supply
    # the OS's already available architecture before importing the extension.
    import platform
    uname = platform.uname()
    uname.__dict__.setdefault('processor', uname.machine)
    import h5py

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with h5py.File(source, 'r') as fh:
        raw = fh.attrs.get('model_config')
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    return scan_config(raw, location + ':model_config')
