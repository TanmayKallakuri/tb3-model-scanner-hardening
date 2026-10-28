"""Inspect Keras configuration data; never instantiate layers or decode code."""
from __future__ import annotations

import json

from modelgate.report import Finding


def scan_config(config, location: str) -> list[Finding]:
    if isinstance(config, (str, bytes)):
        config = json.loads(config)
    findings = []
    todo = [(config, location)]
    while todo:
        value, path = todo.pop()
        if isinstance(value, dict):
            if value.get('class_name') == 'Lambda':
                layer_config = value.get('config', {})
                function = layer_config.get('function') if isinstance(layer_config, dict) else None
                if (isinstance(function, dict) and function.get('class_name') == '__lambda__'
                        and isinstance(function.get('config'), dict)
                        and function['config'].get('code') is not None):
                    findings.append(Finding('code-execution', path, 'Lambda function carries serialized bytecode'))
                # Keras 2 HDF5 also uses [code, defaults, closure]. Named
                # functions use a string/dict and are not serialized bytecode.
                elif isinstance(function, (list, tuple)) and function and isinstance(function[0], str):
                    findings.append(Finding('code-execution', path, 'Lambda function carries serialized bytecode'))
            todo.extend((v, f'{path}.{k}') for k, v in value.items() if isinstance(v, (dict, list)))
        elif isinstance(value, list):
            todo.extend((v, f'{path}[{i}]') for i, v in enumerate(value) if isinstance(v, (dict, list)))
    return findings


def scan_h5(source, location: str) -> list[Finding]:
    import platform

    # h5py indexes platform.uname(), whose lazy `processor` field otherwise
    # launches `uname -p` on Python 3.12/Linux. Kernel-provided machine data is
    # sufficient for its platform checks; importing h5py must not spawn.
    system = platform.uname()
    if hasattr(system, '__dict__'):
        system.__dict__.setdefault('processor', system.machine)
    import h5py

    # An archive member is an in-memory file, never extracted to disk.
    with h5py.File(source, 'r') as fh:
        raw = fh.attrs.get('model_config')
    return [] if raw is None else scan_config(raw, location)
