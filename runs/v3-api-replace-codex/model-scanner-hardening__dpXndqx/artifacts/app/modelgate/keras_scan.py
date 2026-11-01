"""Read Keras configuration as data; never deserialize Keras objects."""
from __future__ import annotations

import io
import json

from modelgate.report import Finding


def _bytecode(value):
    if not isinstance(value, dict) or value.get('class_name') != '__lambda__':
        return False
    cfg = value.get('config', {})
    return isinstance(cfg, dict) and ('code' in cfg or 'value' in cfg)


def scan_config(config, location):
    findings = []
    pending = [(config, location)]
    while pending:
        if len(findings) >= 100: return findings
        obj, loc = pending.pop()
        if isinstance(obj, list):
            pending.extend((v, f'{loc}[{i}]') for i, v in enumerate(obj))
        elif isinstance(obj, dict):
            cfg = obj.get('config')
            if isinstance(cfg, dict) and isinstance(obj.get('class_name'), str):
                fields = ['activation']
                if obj.get('class_name') == 'Lambda':
                    fields += ['function', 'output_shape']
                for field in fields:
                    if _bytecode(cfg.get(field)):
                        findings.append(Finding('code-execution', f'{loc}.config.{field}',
                                                'serialized Python bytecode in Keras ' + field))
                # Keras 2 writes Lambda bytecode as [code, defaults, closure].
                if obj.get('class_name') == 'Lambda':
                    for field in ('function', 'output_shape'):
                        if (cfg.get(field + '_type') == 'lambda' and
                                isinstance(cfg.get(field), (list, tuple))):
                            findings.append(Finding('code-execution', f'{loc}.config.{field}',
                                                    'serialized Python bytecode in Keras ' + field))
            # Model/layer structure is distinct from arbitrary metadata or
            # constants that happen to describe a Lambda as ordinary data.
            for key in ('config', 'layers', 'layer', 'backward_layer'):
                value = obj.get(key)
                if isinstance(value, (dict, list)): pending.append((value, f'{loc}.{key}'))
    return findings


def scan_json(raw, location):
    return scan_config(json.loads(raw), location)


def scan_h5(source, location):
    # h5py's extension asks for platform.uname()[0]. On Python 3.12 even
    # indexing that tuple lazily probes the processor via `uname -p`. Supply
    # the diagnostic field from the in-process uname syscall before importing
    # h5py, so a read-only scan never spawns a helper process.
    import os
    # The scanner never does numerical work; a library import must not create
    # a native worker pool, and one thread also keeps startup predictable.
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '1'
    import platform
    platform.uname().__dict__.setdefault('processor', platform.uname().machine)
    import h5py

    # A BytesIO driver keeps container members in memory, with no extraction.
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with h5py.File(source, 'r') as fh:
        raw = fh.attrs.get('model_config')
    return [] if raw is None else scan_json(raw, location + ':model_config')
