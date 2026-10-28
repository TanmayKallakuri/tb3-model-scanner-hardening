"""Read Keras configuration as data; never import Keras or decode bytecode."""
from __future__ import annotations

import json

from modelgate.report import Finding


def scan_config(config, location: str) -> list[Finding]:
    if isinstance(config, (str, bytes)):
        config = json.loads(config)
    findings = []
    pending = [(config, location)]
    while pending:
        obj, path = pending.pop()
        if isinstance(obj, dict):
            if obj.get('class_name') == 'Lambda':
                cfg = obj.get('config', {})
                fn = cfg.get('function') if isinstance(cfg, dict) else None
                if (isinstance(fn, dict) and fn.get('class_name') == '__lambda__'
                        and isinstance(fn.get('config'), dict) and 'code' in fn['config']):
                    findings.append(Finding('code-execution', path, 'Lambda function contains serialized Python bytecode'))
                # Keras 2's HDF5 representation of the same serialized function.
                elif isinstance(cfg, dict) and cfg.get('function_type') == 'lambda' and isinstance(fn, (list, tuple)) and fn:
                    findings.append(Finding('code-execution', path, 'Lambda function contains serialized Python bytecode'))
            pending.extend((v, f'{path}.{k}') for k, v in obj.items() if isinstance(v, (dict, list)))
        elif isinstance(obj, list):
            pending.extend((v, f'{path}[{i}]') for i, v in enumerate(obj) if isinstance(v, (dict, list)))
    return findings


def scan_h5(source, location: str) -> list[Finding]:
    # Only metadata is read. No weights, links, or model objects are loaded.
    import h5py

    with h5py.File(source, 'r') as fh:
        raw = fh.attrs.get('model_config')
    return [] if raw is None else scan_config(raw, f'{location}:model_config')
