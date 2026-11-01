"""Read Keras configuration data without deserializing any Keras objects."""
from __future__ import annotations

import io
import json
from modelgate.report import Finding


def _bytecode(value):
    if not isinstance(value, dict) or value.get('class_name') != '__lambda__':
        return False
    cfg = value.get('config', {})
    return isinstance(cfg, dict) and (
        isinstance(cfg.get('code'), str) or
        (isinstance(cfg.get('value'), (list, tuple)) and bool(cfg['value'])
         and isinstance(cfg['value'][0], str)))


def scan_config(config, location: str) -> list[Finding]:
    if isinstance(config, (bytes, str)):
        config = json.loads(config)
    findings = []
    todo = [(config, location)]
    while todo:
        value, loc = todo.pop()
        if isinstance(value, dict):
            cfg = value.get('config')
            if isinstance(cfg, dict) and isinstance(value.get('class_name'), str):
                fields = ['activation']
                if value['class_name'] == 'Lambda':
                    fields += ['function', 'output_shape']
                for field in fields:
                    fn = cfg.get(field)
                    legacy = (value['class_name'] == 'Lambda' and field != 'activation'
                              and isinstance(fn, list) and len(fn) == 3
                              and isinstance(fn[0], str)
                              and cfg.get(field + '_type') == 'lambda')
                    if _bytecode(fn) or legacy:
                        findings.append(Finding('code-execution', f'{loc}.config.{field}',
                                                'serialized Python bytecode in Keras callable'))
            cls=value.get('class_name')
            if cls=='function' and isinstance(cfg,str):
                continue  # unused fields on a named reference are data
            if isinstance(cls,str) and isinstance(cfg,(dict,list)):
                if cls in ('Sequential','Functional','Model') and isinstance(cfg,dict):
                    # These model loaders rebuild their layers; extra model
                    # metadata is not passed to the layer deserializer.
                    layers=cfg.get('layers')
                    if isinstance(layers,(dict,list)): todo.append((layers,f'{loc}.config.layers'))
                else:
                    todo.append((cfg,f'{loc}.config'))
            else:
                todo.extend((v, f'{loc}.{k}') for k, v in value.items() if isinstance(v, (dict, list)))
        elif isinstance(value, list):
            todo.extend((v, f'{loc}[{i}]') for i, v in enumerate(value) if isinstance(v, (dict, list)))
    return findings


def scan_h5(source, location: str) -> list[Finding]:
    # h5py's version metadata iterates platform.uname(). On CPython that
    # lazily asks `uname -p` via subprocess. Seed the optional processor field
    # with the already available kernel machine value, so importing our parser
    # cannot start a process (or open /dev/null for that subprocess).
    import platform
    uname = platform.uname()
    uname.processor = uname.machine
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
