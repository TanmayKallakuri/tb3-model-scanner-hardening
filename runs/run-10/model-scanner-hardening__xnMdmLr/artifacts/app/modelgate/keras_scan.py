"""Inspect Keras configuration data; never deserialize a Keras object."""
from __future__ import annotations

import json
from modelgate.report import Finding


def bytecode(value):
    if not isinstance(value, dict) or value.get('class_name') != '__lambda__':
        return False
    config = value.get('config', {})
    if not isinstance(config, dict):
        return False
    return isinstance(config.get('code'), str) or (
        isinstance(config.get('value'), (list, tuple)) and
        bool(config['value']) and isinstance(config['value'][0], str))


def scan_config(config, location: str) -> list[Finding]:
    if isinstance(config, (str, bytes)):
        config = json.loads(config)
    findings = []
    pending = [(config, location)]
    while pending:
        value, where = pending.pop()
        if isinstance(value, dict):
            cls = value.get('class_name')
            cfg = value.get('config')
            if isinstance(cls, str) and isinstance(cfg, dict):
                if cls == 'Lambda':
                    for field in ('function', 'output_shape'):
                        func = cfg.get(field)
                        legacy = (cfg.get(field + '_type') == 'lambda' and
                                  isinstance(func, (list, tuple)) and bool(func) and
                                  isinstance(func[0], str))
                        if bytecode(func) or legacy:
                            findings.append(Finding('code-execution', f'{where}/config/{field}',
                                                    f'Lambda {field} contains serialized bytecode'))
                if bytecode(cfg.get('activation')):
                    findings.append(Finding('code-execution', f'{where}/config/activation',
                                            'activation contains serialized bytecode'))
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    pending.append((child, f'{where}/{key}'))
        elif isinstance(value, list):
            pending.extend((child, f'{where}[{i}]') for i, child in enumerate(value))
    return findings


def scan_h5(path_or_buffer, location: str) -> list[Finding]:
    # h5py queries platform.uname()[0]. On Python 3.12 indexing that
    # tuple also resolves processor via a `uname -p` subprocess. Supply the
    # syscall-only answer up front so dependency initialization is read-only.
    import os
    import platform
    uname = platform.uname_result(*os.uname())
    uname.__dict__['processor'] = ''
    platform._uname_cache = uname
    import h5py
    with h5py.File(path_or_buffer, 'r') as fh:
        raw = fh.attrs.get('model_config')
    return [] if raw is None else scan_config(raw, location + ':model_config')
