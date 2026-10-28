"""Build a reproducible 650-artifact policy stress corpus, without loading it.

Usage: python3 -B modelgate/tests/build_corpus.py /tmp/modelgate-corpus
The resulting directory can be passed to evaluate.py --samples.
"""
import base64
import bz2
import gzip
import io
import json
import lzma
import marshal
import pathlib
import pickle
import sys
import tarfile
import zlib
from test_semantics import *
from modelgate.tfproto import saved_model_pb2


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    bad = stream(call(glob('os', 'system'), 'unexecuted-command'))
    good = stream({'comment': 'os.system eval( subprocess'})
    source = call(glob('io', 'BytesIO'), bad)
    template_pairs = [
        (call(glob('os', 'system'), 'unexecuted-command'), glob('os', 'system')),
        (call(glob('builtins', 'eval'), '1+1'), call(glob('builtins', 'str'), 'eval(1+1)')),
        (call(glob('builtins', 'open'), '/outside'), call(glob('os', 'fspath'), '/outside')),
        (method(call(glob('socket', 'socket')), 'connect', ('example.invalid', 443)), call(glob('socket', 'socket'))),
        (call(glob('ftplib', 'FTP'), 'example.invalid'), call(glob('ftplib', 'FTP'))),
        (method(call(glob('http.client', 'HTTPConnection'), 'example.invalid'), 'request', 'GET', '/'), call(glob('http.client', 'HTTPConnection'), 'example.invalid')),
        (call(glob('subprocess', 'Popen'), ['command']), Expr(lit(glob('subprocess', 'Popen')) + lit((['command'],)) + b'\x81')),
        (call(partial(glob('os', 'system'), 'x')), partial(glob('os', 'system'), 'x')),
        (call(attr(module('os'), 'system'), 'x'), attr(module('os'), 'system')),
        (call(call(call(glob('operator', 'attrgetter'), 'os.system'), module('shutil')), 'x'), call(call(glob('operator', 'attrgetter'), 'os.system'), module('shutil'))),
        (call(call(glob('operator', 'getitem'), call(glob('builtins', 'vars'), module('os')), 'system'), 'x'), call(glob('builtins', 'vars'), module('os'))),
        (call(glob('builtins', 'list'), call(glob('builtins', 'map'), glob('os', 'system'), ['x'])), call(glob('builtins', 'map'), glob('os', 'system'), ['x'])),
        (call(glob('builtins', 'list'), call(glob('builtins', 'filter'), glob('os', 'system'), ['x'])), call(glob('builtins', 'filter'), glob('os', 'system'), ['x'])),
        (call(glob('builtins', 'list'), call(glob('itertools', 'starmap'), glob('os', 'system'), [('x',)])), call(glob('itertools', 'starmap'), glob('os', 'system'), [('x',)])),
        (call(partial(glob('builtins', 'sorted'), key=glob('os', 'system')), ['x']), partial(glob('builtins', 'sorted'), key=glob('os', 'system'))),
        (call(glob('pickle', 'loads'), bad), call(glob('pickle', 'loads'), good + bad)),
        (call(glob('pickle', 'loads'), call(glob('base64', 'b64decode'), base64.b64encode(bad))), call(glob('pickle', 'loads'), call(glob('base64', 'b64decode'), base64.b64encode(good)))),
        (call(glob('pickle', 'loads'), call(glob('builtins', 'bytearray'), list(bad))), call(glob('pickle', 'loads'), call(glob('builtins', 'bytearray'), list(good)))),
        (call(glob('pickle', 'loads'), call(glob('zlib', 'decompress'), zlib.compress(bad))), call(glob('pickle', 'loads'), call(glob('zlib', 'decompress'), zlib.compress(good)))),
        (call(call(glob('pkgutil', 'resolve_name'), 'shutil.os.system'), 'x'), call(glob('pkgutil', 'resolve_name'), 'shutil.os.system')),
        (call(call(call(glob('functools', 'lru_cache'), None), glob('os', 'system')), 'x'), call(call(glob('functools', 'lru_cache'), None), glob('os', 'system'))),
        (method(call(partial(glob('threading', 'Thread'), target=glob('os', 'system'), args=('x',))), 'start'), call(partial(glob('threading', 'Thread'), target=glob('os', 'system'), args=('x',)))),
        (method(call(glob('concurrent.futures', 'ThreadPoolExecutor')), 'submit', glob('os', 'system'), 'x'), method(call(glob('concurrent.futures', 'ThreadPoolExecutor')), 'submit', glob('builtins', 'str'), 'x')),
        (call(partial(glob('numpy', 'load'), allow_pickle=True), source), call(glob('numpy', 'load'), source)),
        (call(glob('logging.config', 'dictConfig'), {'version': 1, 'handlers': {'f': {'class': 'logging.FileHandler', 'filename': '/outside'}}}), call(glob('logging.config', 'dictConfig'), {'version': 1, 'handlers': {'f': {'class': 'logging.FileHandler', 'filename': '/outside', 'delay': True}}})),
        (call(glob('types', 'FunctionType'), call(glob('marshal', 'loads'), marshal.dumps((lambda x: x).__code__)), {}), call(glob('marshal', 'loads'), marshal.dumps((lambda x: x).__code__))),
    ]

    def tar(data):
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode='w') as tf:
            info = tarfile.TarInfo('export/model.unknown')
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        return out.getvalue()

    wrappers = [lambda x: x, gzip.compress, bz2.compress, lzma.compress,
                lambda x: archive([('model.odd', x)], prefix=b'archive-stub\n'),
                lambda x: archive([('old.pkl', good), ('model.pkl', x), ('card.json', b'{}')]),
                tar, lambda x: gzip.compress(tar(x)),
                lambda x: npy(x), lambda x: archive([('array.npy', npy(x))]),
                lambda x: archive([('weights/data.pkl', x), ('weights/version', b'3\n'), ('weights/data/0', b'raw storage')]),
                lambda x: gzip.compress(bz2.compress(archive([('model', x)])))]
    cases = []
    for i, (malicious, clean) in enumerate(template_pairs):
        for j, wrap in enumerate(wrappers):
            for expr, expected in [(malicious, 'malicious'), (clean, 'clean')]:
                cases.append((wrap(stream(expr, proto=4)), expected))
    # 26 pairs x 12 wrappers x 2 verdicts = 624 cases.
    for i in range(13):
        for dangerous in (True, False):
            sm = saved_model_pb2.SavedModel()
            meta = sm.meta_graphs.add()
            if i % 2:
                meta = sm.meta_graphs.add()
            nodes = meta.graph_def.library.function.add().node_def if i % 3 else meta.graph_def.node
            node = nodes.add()
            node.name = 'ReadFile'
            node.op = ['PyFunc', 'WriteFile', 'DataServiceDatasetV4'][i % 3] if dangerous else ['SaveV2', 'Const', 'Identity'][i % 3]
            cases.append((archive([('model/saved_model.pb', sm.SerializeToString()), ('model/variables/variables.index', b'index')]), 'malicious' if dangerous else 'clean'))
    assert len(cases) == 650
    labels = {}
    for i, (raw, expected) in enumerate(cases):
        name = f'case_{i:04d}.bin'
        (directory / name).write_bytes(raw)
        labels[name] = expected
    (directory / 'labels.json').write_text(json.dumps(labels))
    print(f'Wrote {len(cases)} artifacts to {directory}')


if __name__ == '__main__':
    build(pathlib.Path(sys.argv[1]))
