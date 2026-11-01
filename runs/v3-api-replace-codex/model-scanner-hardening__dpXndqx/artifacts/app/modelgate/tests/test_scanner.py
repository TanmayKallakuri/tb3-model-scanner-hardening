"""Policy regression cases. Generated pickles are ONLY statically scanned.

The emitter below serializes opcode descriptions, never Python __reduce__
objects, so neither test generation nor validation invokes artifact behavior.
"""
import bz2
import gzip
import io
import json
import lzma
import os
import pickle
import struct
import sys
import tarfile
import tempfile
import unittest
import zipfile
import zlib
from dataclasses import dataclass
from unittest import mock

from modelgate import dispatch, keras_scan, savedmodel_scan
from modelgate.pickle_scan import Interpreter, TORCH_MAGIC, interpret
from modelgate.symbolic import LoadError, World


@dataclass
class Global:
    module: str
    name: str


@dataclass
class Call:
    target: object
    args: object = ()
    op: bytes = b'R'


def emit(obj):
    if isinstance(obj, Global): return b'c' + obj.module.encode() + b'\n' + obj.name.encode() + b'\n'
    if isinstance(obj, Call): return emit(obj.target) + emit(obj.args) + obj.op
    if obj is None: return b'N'
    if obj is True: return b'\x88'
    if obj is False: return b'\x89'
    if isinstance(obj, str):
        data = obj.encode('utf-8')
        return b'X' + struct.pack('<I', len(data)) + data
    if isinstance(obj, bytes): return b'B' + struct.pack('<I', len(obj)) + obj
    if isinstance(obj, bytearray): return b'\x96' + struct.pack('<Q', len(obj)) + obj
    if isinstance(obj, int):
        if -2**31 <= obj < 2**31: return b'J' + struct.pack('<i', obj)
        return b'L' + str(obj).encode() + b'L\n'
    if isinstance(obj, tuple): return b'(' + b''.join(emit(v) for v in obj) + b't'
    if isinstance(obj, list): return b'(' + b''.join(emit(v) for v in obj) + b'l'
    if isinstance(obj, dict): return b'(' + b''.join(emit(k) + emit(v) for k, v in obj.items()) + b'd'
    raise TypeError(type(obj))


def stream(obj, proto=4): return b'\x80' + bytes([proto]) + emit(obj) + b'.'


def global_(name):
    module, attr = name.rsplit('.', 1)
    return Global(module, attr)


def call(name, *args): return Call(global_(name), args)


def indirect_get(obj, attr): return call('builtins.getattr', obj, attr)


def scan(data, location='artifact'):
    world = World()
    world.embedded = lambda loader, raw, loc, kw: dispatch._embedded(world, loader, raw, loc, kw)
    dispatch.scan_data(data, location, world)
    return world.findings


def zip_bytes(entries, prefix=b''):
    out = io.BytesIO()
    out.write(prefix)
    with zipfile.ZipFile(out, 'w') as zf:
        for name, data in entries: zf.writestr(name, data)
    return out.getvalue()


def npy(payload, descr='|O', version=1):
    header = repr({'descr': descr, 'fortran_order': False, 'shape': (1,)}) + '\n'
    encoded = header.encode()
    return b'\x93NUMPY' + bytes([version, 0]) + struct.pack('<H' if version == 1 else '<I', len(encoded)) + encoded + payload


class PicklePolicy(unittest.TestCase):
    def assertVerdict(self, obj, malicious, proto=4):
        findings = scan(stream(obj, proto))
        self.assertEqual(bool(findings), malicious, findings)
        self.assertFalse(any(f.kind == 'unsupported' for f in findings), findings)

    def test_data_is_inert(self):
        for obj in (global_('os.system'), ['os.system', 'subprocess', 'eval(1)'],
                    {'stored': global_('builtins.eval')}, call('builtins.__import__', 'os'),
                    call('operator.attrgetter', 'os.system'), call('functools.partial', global_('os.system'), 'unused')):
            with self.subTest(obj=obj): self.assertVerdict(obj, False)

    def test_call_and_aliases(self):
        for module, name, args in (
                ('os', 'system', ('unused',)), ('posix', 'system', ('unused',)),
                ('nt', 'system', ('unused',)), ('__builtin__', 'eval', ('1+1',)),
                ('commands', 'getoutput', ('unused',)), ('urllib2', 'urlopen', ('https://example.invalid',)),
                ):
            with self.subTest(module=module, name=name):
                self.assertVerdict(Call(Global(module, name), args), True, 2 if module in ('__builtin__', 'commands', 'urllib2') else 4)

        self.assertVerdict(Call(indirect_get(Call(Global('httplib', 'HTTPConnection'), ('example.invalid',)), 'request'), ('GET', '/')), True, 2)

    def test_protocol_zero_inst_obj(self):
        cases = [b"(V1+1\nibuiltins\neval\n.", b"(cbuiltins\neval\nV1+1\no."]
        for data in cases: self.assertTrue(scan(data))
        for data in (b'(iftplib\nFTP\n.', b'(cftplib\nFTP\no.'):
            self.assertEqual(scan(data), [])

    def test_call_routes(self):
        imported = call('builtins.__import__', 'os')
        routes = [
            indirect_get(imported, 'system'),
            Call(call('operator.attrgetter', 'system'), (imported,)),
            call('functools.partial', global_('os.system')),
            Call(call('operator.methodcaller', 'system', 'unused'), (imported,)),
            call('pkgutil.resolve_name', 'os:system'),
            indirect_get(call('importlib.import_module', 'shutil'), 'os'),
            call('operator.getitem', indirect_get(call('importlib.import_module', 'sys'), 'modules'), 'os'),
        ]
        for i, route in enumerate(routes):
            if i == 3: obj = route
            elif i in (5, 6): obj = Call(indirect_get(route, 'system'), ('unused',))
            else: obj = Call(route, ('unused',))
            with self.subTest(i=i): self.assertVerdict(obj, True)
        self.assertVerdict(Call(call('operator.getitem', call('builtins.vars', imported), 'system'), ('unused',)), True)

    def test_wrapped_callables(self):
        target = global_('builtins.eval')
        for route in (call('copy.copy', target), call('copy.deepcopy', target),
                      call('functools.lru_cache', target), call('functools.cache', target),
                      Call(call('functools.lru_cache', 16), (target,))):
            self.assertVerdict(route, False)
            self.assertVerdict(Call(route, ('1+1',)), True)

    def test_invalid_calls_stop(self):
        for obj in (Call(global_('os.system'), ['unused']), Call(global_('builtins.eval'), []),
                    call('os.system'), call('builtins.eval'),
                    Call(global_('os.system'), ('unused',), b'\x81')):
            with self.subTest(obj=obj): self.assertVerdict(obj, False)
        # Invalid dotted lookups below protocol 4 never reach REDUCE.
        self.assertVerdict(Call(Global('bdb', 'Bdb.run'), ('1+1',)), False, 2)
        data = b'Pstorage\n' + emit(call('builtins.eval', '1+1')) + b'.'
        self.assertEqual(scan(data), [])

    def test_newobj_vs_constructor(self):
        for name, args in (('subprocess.Popen', ('unused',)), ('io.FileIO', ('unused',)),
                           ('smtplib.SMTP', ('example.invalid',))):
            self.assertVerdict(Call(global_(name), args, b'\x81'), False)
            self.assertVerdict(Call(global_(name), args), True)
        for name in ('socket.socket', 'ftplib.FTP', 'smtplib.SMTP', 'code.InteractiveInterpreter'):
            self.assertVerdict(call(name), False)
        self.assertVerdict(call('http.client.HTTPConnection', 'example.invalid', 80), False)
        self.assertVerdict(Call(indirect_get(call('http.client.HTTPConnection', 'example.invalid'), 'connect')), True)
        self.assertVerdict(Call(indirect_get(call('socket.socket'), 'connect'), (('example.invalid', 80),)), True)

    def test_memo_stack_and_partial_retarget(self):
        target = emit(global_('builtins.eval'))
        for put, get in ((b'p0\n', b'g0\n'), (b'q\x00', b'h\x00'), (b'r\x00\x00\x00\x00', b'j\x00\x00\x00\x00'), (b'\x94', b'h\x00')):
            data = b'\x80\x04' + target + put + b'0' + get + emit(('1+1',)) + b'R.'
            self.assertTrue(scan(data))
        partial = call('functools.partial', global_('builtins.str'))
        state = (global_('builtins.eval'), ('1+1',), {}, None)
        prefix = b'\x80\x04' + emit(partial) + b'q\x00' + emit(state) + b'b'
        self.assertEqual(scan(prefix + b'.'), [])
        self.assertTrue(scan(prefix + b'0h\x00)R.'))

    def test_module_build_and_later_streams(self):
        module = emit(call('importlib.import_module', 'operator'))
        first = b'\x80\x04' + module + emit({'innocent': global_('builtins.eval')}) + b'b.'
        self.assertEqual(scan(first), [])
        self.assertTrue(scan(first + stream(Call(Global('operator', 'innocent'), ('1+1',)))))
        self.assertTrue(scan(stream({'weights': [1, 2, 3]}) + stream(call('builtins.eval', '1+1'))))

    def test_lazy_consumption(self):
        for name in ('builtins.map', 'builtins.filter', 'itertools.starmap'):
            vals = [('1+1',)] if name.endswith('starmap') else ['1+1']
            lazy = call(name, global_('builtins.eval'), vals)
            self.assertVerdict(lazy, False)
            for consumer in ('builtins.list', 'builtins.tuple', 'builtins.next', 'collections.deque'):
                with self.subTest(name=name, consumer=consumer): self.assertVerdict(call(consumer, lazy), True)
        self.assertVerdict(call('builtins.list', call('builtins.map', global_('builtins.eval'), [])), False)
        self.assertVerdict(call('builtins.list', call('builtins.map', global_('builtins.str'), [1, 2])), False)

    def test_callback_consumers(self):
        for name in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            partial = call('functools.partial', global_(name))
            state = (global_(name), (), {'key': global_('builtins.eval')}, None)
            data = b'\x80\x04' + emit(partial) + emit(state) + b'b' + emit((['1+1'],)) + b'R.'
            self.assertTrue(scan(data))
        self.assertVerdict(call('operator.call', global_('builtins.eval'), '1+1'), True)
        self.assertVerdict(call('functools.reduce', global_('builtins.eval'), ['1+1']), False)
        self.assertVerdict(call('functools.reduce', global_('builtins.eval'), ['1+1', {}]), True)

    def test_nested_loader_reads_only_what_it_loads(self):
        clean = stream({'weights': [1, 2, 3]})
        bad = stream(call('builtins.eval', '1+1'))
        for raw, expected in ((clean, False), (bad, True), (clean + bad, False), (bad + clean, True)):
            for wrapper in (lambda b: b, lambda b: bytearray(b),
                            lambda b: call('base64.b64decode', __import__('base64').b64encode(b)),
                            lambda b: call('zlib.decompress', zlib.compress(b)),
                            lambda b: Call(Global('builtins', 'bytes.fromhex'), (b.hex(),)),
                            lambda b: call('operator.add', b[:8], b[8:])):
                with self.subTest(expected=expected, wrapper=wrapper):
                    self.assertVerdict(call('pickle.loads', wrapper(raw)), expected)
            self.assertVerdict(call('pickle.load', call('io.BytesIO', raw)), expected)
        self.assertVerdict(call('pickle.load', 'outside.pkl'), True)

    def test_torch_legacy_all_five_streams(self):
        clean = stream({'weights': [1, 2, 3]})
        bad = stream(call('builtins.eval', '1+1'))
        for i in range(1, 5):
            streams = [stream(TORCH_MAGIC), stream(1001), stream({}), clean, stream([])]
            streams[i] = bad
            self.assertTrue(scan(b''.join(streams)))
        self.assertEqual(scan(b''.join([stream(TORCH_MAGIC), stream(1001), stream({}), clean, stream([])]) + b'not pickle'), [])

    def test_all_protocols_for_plain_data(self):
        # pickle.dumps only handles built-in data here; no artifact callables.
        for proto in range(6):
            self.assertEqual(scan(pickle.dumps({'weights': [1, 2, 3], 'notes': 'os.system'}, protocol=proto)), [])

    def test_pickled_memory_stream_state(self):
        raw = stream(call('builtins.eval', '1+1'))
        # This is the state shape emitted by io.BytesIO's own pickler.
        memory = emit(call('io.BytesIO')) + emit((raw, 0, None)) + b'b'
        data = b'\x80\x04' + emit(global_('pickle.load')) + b'(' + memory + b'tR.'
        self.assertTrue(scan(data))
        safe = emit(call('io.BytesIO')) + emit((stream({'data': 1}), 0, None)) + b'b'
        self.assertEqual(scan(b'\x80\x04' + emit(global_('pickle.load')) + b'(' + safe + b'tR.'), [])

    def test_marshal_function_materialization(self):
        import marshal
        # The test's own harmless code object supplies valid serialized code;
        # the scanner keeps it as a symbol and never calls marshal.loads.
        raw = marshal.dumps((lambda: None).__code__)
        code = call('marshal.loads', raw)
        self.assertVerdict(code, False)
        for op in (b'R', b'\x81'):
            self.assertVerdict(Call(Global('types', 'FunctionType'), (code, {}), op), True)
        self.assertVerdict(call('marshal.loads', marshal.dumps(123)), False)

    def test_thread_and_executor_callbacks(self):
        for target, expected in ((global_('builtins.eval'), True), (global_('builtins.str'), False)):
            thread_ctor = call('functools.partial', global_('threading.Thread'))
            state = (global_('threading.Thread'), (), {'target': target, 'args': ('1+1',)}, None)
            prefix = b'\x80\x04' + emit(thread_ctor) + emit(state) + b'b)R'
            self.assertEqual(scan(prefix + b'.'), [])
            # Memoize the constructed thread, obtain its start method, call it.
            data = prefix + b'q\x00' + emit(global_('builtins.getattr')) + b'(h\x00' + emit('start') + b'tR)R.'
            self.assertEqual(bool(scan(data)), expected)
            executor = call('concurrent.futures.ThreadPoolExecutor')
            self.assertVerdict(Call(indirect_get(executor, 'submit'), (target, '1+1')), expected)
            self.assertVerdict(Call(Global('concurrent.futures', 'ThreadPoolExecutor.submit'), (executor, target, '1+1')), expected)
        self.assertVerdict(call('concurrent.futures.ProcessPoolExecutor'), False)

    def test_short_circuit_and_empty_consumers(self):
        bad = call('builtins.map', global_('builtins.eval'), ['1+1'])
        for name in ('builtins.map', 'builtins.zip'):
            lazy = call(name, global_('builtins.str'), [], bad) if name.endswith('map') else call(name, [], bad)
            self.assertVerdict(call('builtins.list', lazy), False)
        self.assertVerdict(call('builtins.any', [True, bad]), False)
        self.assertVerdict(call('builtins.list', call('itertools.islice', bad, 0)), False)

    def test_newobj_ex_partial_and_protocol_zero_memos(self):
        data = b'\x80\x04' + emit(global_('functools.partial')) + emit((global_('builtins.eval'),)) + emit({}) + b'\x92' + emit(('1+1',)) + b'R.'
        self.assertTrue(scan(data))
        text = b'cbuiltins\neval\np42\n0g42\n(V1+1\ntR.'
        self.assertTrue(scan(text))

    def test_single_nested_stream_never_infers_legacy(self):
        raw = stream(TORCH_MAGIC) + stream(call('builtins.eval', '1+1'))
        self.assertVerdict(call('pickle.loads', raw), False)
        self.assertEqual(scan(npy(raw)), [])
        self.assertVerdict(call('torch.load', call('io.BytesIO', stream(123) + stream(call('builtins.eval', '1+1')))), False)

    def test_timeit_and_bdb_callable_arguments(self):
        for target, expected in ((global_('builtins.str'), False), (call('functools.partial', global_('builtins.eval'), '1+1'), True)):
            self.assertVerdict(call('timeit.timeit', target), expected)
            self.assertVerdict(call('timeit.Timer', target), False)
            self.assertVerdict(Call(indirect_get(call('timeit.Timer', target), 'timeit')), expected)
            self.assertVerdict(Call(indirect_get(call('bdb.Bdb'), 'runcall'), (target,)), expected)
        self.assertVerdict(call('timeit.timeit', 'pass'), True)
        self.assertVerdict(call('os.fspath', call('pathlib.Path', 'inside')), False)

    def test_namespace_copies_and_function_globals(self):
        module = call('importlib.import_module', 'os')
        namespace = Call(indirect_get(call('builtins.vars', module), 'copy'))
        target = call('operator.getitem', namespace, 'system')
        self.assertVerdict(Call(target, ('unused',)), True)
        run = Global('bdb', 'Bdb.run')
        globs = indirect_get(run, '__globals__')
        builtins = call('operator.getitem', globs, '__builtins__')
        self.assertVerdict(Call(call('operator.getitem', builtins, 'eval'), ('1+1',)), True)

    def test_partial_state_preserves_empty_keyword_dictionary(self):
        # BUILD stores this exact dictionary. A later mutation must be visible
        # when the memoized partial is finally called.
        prefix = b'\x80\x04' + emit(call('functools.partial', global_('builtins.str')))
        state = b'(' + emit(global_('subprocess.Popen')) + b')}' + b'q\x00Nt'
        data = prefix + state + b'bq\x010h\x00' + emit('args') + emit('unused') + b's0h\x01)R.'
        self.assertTrue(scan(data))

    def test_safe_serialization_and_defaultdict_factories(self):
        self.assertVerdict(call('pickle.loads', call('pickle.dumps', {'weights': [1, 2, 3]})), False)
        for factory, expected in ((global_('builtins.str'), False), (call('functools.partial', global_('builtins.eval'), '1+1'), True)):
            obj = call('collections.defaultdict', factory)
            self.assertVerdict(obj, False)
            self.assertVerdict(call('operator.getitem', obj, 'new_key'), expected)
        self.assertVerdict(Call(call('builtins.getattr', 'data', 'absent', global_('builtins.eval')), ('1+1',)), True)

    def test_inst_obj_zero_args_partial_is_still_called(self):
        partial = emit(call('functools.partial', global_('builtins.eval'), '1+1'))
        self.assertTrue(scan(b'\x80\x04(' + partial + b'o.'))
        # A callable rebound onto a module is still a function, even when INST
        # receives no arguments, and must not be treated as a class.
        module = emit(call('importlib.import_module', 'operator'))
        first = b'\x80\x04' + module + b'(' + emit('run') + partial + b'db.'
        self.assertTrue(scan(first + b'(ioperator\nrun\n.'))

    def test_dynamic_types_preserve_constructor_capabilities(self):
        base = global_('builtins.object')
        pure = call('builtins.type', 'Container', (base,), {'value': global_('builtins.eval')})
        self.assertVerdict(Call(pure), False)
        execute = call('builtins.type', 'Runner', (base,), {'__init__': global_('os.system')})
        self.assertVerdict(execute, False)
        self.assertVerdict(Call(execute, ('unused',)), True)
        derived = call('builtins.type', 'Runner', (global_('subprocess.Popen'),), {})
        self.assertVerdict(Call(derived, ('unused',)), True)
        self.assertVerdict(Call(derived, ('unused',), b'\x81'), False)
        overridden = call('builtins.type', 'Runner', (global_('subprocess.Popen'),), {'__init__': global_('builtins.str')})
        self.assertVerdict(Call(overridden, ('unused',)), False)
        tuple_type = call('builtins.type', 'Args', (global_('builtins.tuple'),), {})
        self.assertVerdict(Call(global_('builtins.eval'), Call(tuple_type, (('1+1',),))), True)

    def test_lazy_wrappers_and_additional_consumers(self):
        for target, expected in ((global_('builtins.str'), False), (global_('builtins.eval'), True)):
            lazy = call('builtins.map', target, ["['x']"])
            joined = Call(indirect_get('', 'join'), (lazy,))
            self.assertVerdict(joined, expected)
            self.assertVerdict(call('collections.Counter', lazy), expected)
            wrapped = Call(Global('itertools', 'chain.from_iterable'), (lazy,))
            self.assertVerdict(wrapped, False)
            self.assertVerdict(call('builtins.list', wrapped), expected)
            tee = call('itertools.tee', lazy)
            self.assertVerdict(tee, False)
            self.assertVerdict(call('builtins.list', call('operator.getitem', tee, 0)), expected)
        self.assertVerdict(call('builtins.reversed', call('builtins.map', global_('builtins.eval'), ['1+1'])), False)

    def test_iterator_observes_memoized_list_mutation(self):
        for name, expected in (('builtins.str', False), ('builtins.eval', True)):
            data = b'\x80\x04' + emit(global_('builtins.map')) + b'(' + emit(global_(name)) + b']q\x00tRq\x010h\x00' + emit('1+1') + b'a0' + emit(global_('builtins.list')) + b'(h\x01tR.'
            self.assertEqual(bool(scan(data)), expected)

    def test_incremental_consumers_stop_before_later_callbacks(self):
        lazy = call('builtins.map', global_('builtins.eval'), ['1+1'])
        for consumer, first in (('builtins.bytes', 'x'), ('builtins.bytearray', 'x'), ('builtins.set', []), ('builtins.frozenset', []), ('builtins.dict', 1)):
            source = call('itertools.chain', [first], lazy)
            with self.subTest(consumer=consumer): self.assertVerdict(call(consumer, source), False)
        source = call('itertools.chain', ['1+1'], 123)
        for consumer, expected in (('builtins.sorted', False), ('builtins.min', True), ('builtins.max', True)):
            partial = call('functools.partial', global_(consumer))
            state = (global_(consumer), (), {'key': global_('builtins.eval')}, None)
            data = b'\x80\x04' + emit(partial) + emit(state) + b'b' + emit((source,)) + b'R.'
            self.assertEqual(bool(scan(data)), expected)

    def test_zip_strict_checks_later_iterator(self):
        lazy = call('builtins.map', global_('builtins.eval'), ['1+1'])
        for strict in (False, True):
            state = (global_('builtins.zip'), (), {'strict': strict}, None)
            data = b'\x80\x04' + emit(call('functools.partial', global_('builtins.zip'))) + emit(state) + b'b' + emit(([], lazy)) + b'R'
            # Build the zip, then force its first next().
            data = b'\x80\x04' + emit(global_('builtins.list')) + b'(' + data[2:] + b'tR.'
            self.assertEqual(bool(scan(data)), strict)

    def test_empty_and_invalid_iterator_construction(self):
        self.assertVerdict(call('builtins.list', call('builtins.zip')), False)
        for constructor, args in (('builtins.map', (global_('builtins.eval'),)), ('builtins.filter', ()), ('itertools.cycle', (123,)), ('itertools.repeat', (1, 'bad count'))):
            # The standard constructor fails, so a later stream is unreachable.
            data = stream(call(constructor, *args)) + stream(call('builtins.eval', '1+1'))
            self.assertEqual(scan(data), [])

    def test_iterator_work_is_bounded(self):
        findings = scan(stream(call('builtins.list', call('itertools.count'))))
        self.assertTrue(findings)
        self.assertEqual(findings[0].kind, 'unsupported')

    def test_nested_torch_uses_local_magic_not_the_zip_directory(self):
        clean_zip = zip_bytes([('a/data.pkl', stream({'data': 1})), ('a/version', b'3\n')])
        bad_zip = zip_bytes([('a/data.pkl', stream(call('builtins.eval', '1+1'))), ('a/version', b'3\n')])
        unsafe_prefix = stream(call('builtins.eval', '1+1'))
        safe_prefix = stream({'data': 1})
        self.assertVerdict(call('torch.load', call('io.BytesIO', unsafe_prefix + clean_zip)), True)
        self.assertVerdict(call('torch.load', call('io.BytesIO', safe_prefix + bad_zip)), False)
        self.assertTrue(scan(safe_prefix + bad_zip))

    def test_nested_pandas_compression_requires_an_explicit_method(self):
        bad = gzip.compress(stream(call('builtins.eval', '1+1')))
        self.assertVerdict(call('pandas.read_pickle', call('io.BytesIO', bad)), False)
        state = (global_('pandas.read_pickle'), (), {'compression': 'gzip'}, None)
        data = b'\x80\x04' + emit(call('functools.partial', global_('pandas.read_pickle'))) + emit(state) + b'b' + emit((call('io.BytesIO', bad),)) + b'R.'
        self.assertTrue(scan(data))

    def test_dictionary_views_keep_live_values(self):
        # Capture dict.values(), then replace a callable in the memoized dict.
        for target, expected in ((global_('builtins.str'), False), (call('functools.partial', global_('builtins.eval'), '1+1'), True)):
            data = b'\x80\x04' + emit({'fn': global_('builtins.str')}) + b'q\x000' + emit(global_('builtins.getattr')) + b'(h\x00' + emit('values') + b'tR)Rq\x010h\x00' + emit('fn') + emit(target) + b's0' + emit(global_('builtins.list')) + b'(' + emit(global_('builtins.map')) + b'(' + emit(global_('operator.call')) + b'h\x01tRtR.'
            self.assertEqual(bool(scan(data)), expected)

    def test_class_signatures_and_static_descriptors(self):
        self.assertVerdict(call('http.client.HTTPConnection'), False)
        method = call('functools.partialmethod', global_('builtins.eval'), '1+1')
        self.assertVerdict(method, False)
        self.assertVerdict(Call(method), False)
        for target, expected in ((global_('builtins.str'), False), (global_('builtins.eval'), True)):
            self.assertVerdict(Call(call('builtins.staticmethod', target), ('1+1',)), expected)
        state = (global_('subprocess.Popen'), (), {'unknown_argument': 'unused'}, None)
        data = b'\x80\x04' + emit(call('functools.partial', global_('subprocess.Popen'))) + emit(state) + b'b)R.'
        self.assertEqual(scan(data), [])

    def test_executor_future_results_carry_callable_identity(self):
        for name, expected in (('str', False), ('eval', True)):
            module = call('importlib.import_module', 'builtins')
            factory = call('functools.partial', global_('builtins.getattr'), module, name)
            executor = call('concurrent.futures.ThreadPoolExecutor')
            future = Call(indirect_get(executor, 'submit'), (factory,))
            result = Call(indirect_get(future, 'result'))
            self.assertVerdict(result, False)
            self.assertVerdict(Call(result, ('1+1',)), expected)
        self.assertVerdict(Call(indirect_get(call('concurrent.futures.ProcessPoolExecutor'), 'submit')), False)

    def test_python2_fix_imports_is_limited_to_older_protocols(self):
        for module, name, args in (('__builtin__', 'eval', ('1+1',)), ('commands', 'getoutput', ('unused',)), ('urllib2', 'urlopen', ('https://example.invalid',))):
            self.assertVerdict(Call(Global(module, name), args), True, 2)
            self.assertVerdict(Call(Global(module, name), args), False, 4)

    def test_partial_flattening_captures_the_original_target(self):
        for namespace in (None, {}):
            for original, retarget, expected_flat in ((global_('builtins.str'), global_('builtins.eval'), False), (global_('builtins.eval'), global_('builtins.str'), True)):
                first_state = (original, ('1+1',), {}, namespace)
                second_state = (retarget, ('1+1',), {}, None)
                # p is memo 0; q=partial(p) is memo 1. Whether q captures p's
                # target or p itself depends on p having an instance dict.
                data = b'\x80\x04' + emit(call('functools.partial', original)) + emit(first_state) + b'bq\x000' + emit(global_('functools.partial')) + b'(h\x00tRq\x010h\x00' + emit(second_state) + b'b0h\x01)R.'
                expected = expected_flat if namespace is None else not expected_flat
                self.assertEqual(bool(scan(data)), expected)

    def test_clean_nested_deserialization_at_depth(self):
        data = stream({'weights': [1, 2, 3]})
        for _ in range(32): data = stream(call('pickle.loads', data))
        self.assertEqual(scan(data), [])

    def test_alias_callables_keep_their_own_module_dictionaries(self):
        bad = stream(call('builtins.eval', '1+1'))
        for shadow_module, target_module, function, args in (
                ('os', 'posix', 'system', ('unused',)),
                ('posix', 'os', 'system', ('unused',)),
                ('pickle', '_pickle', 'loads', (bad,)),
                ('_pickle', 'pickle', 'loads', (bad,)),
                ('io', '_io', 'open', ('outside',)),
                ('_io', 'builtins', 'open', ('outside',))):
            first = b'\x80\x04' + emit(call('importlib.import_module', shadow_module)) + emit({function: global_('builtins.str')}) + b'b.'
            self.assertTrue(scan(first + stream(Call(Global(target_module, function), args))))
        first = b'\x80\x04' + emit(call('importlib.import_module', 'os')) + emit({'system': global_('builtins.str')}) + b'b.'
        self.assertEqual(scan(first + stream(call('os.system', 'unused'))), [])

    def test_module_attribute_deletion_stops_a_later_global(self):
        module = call('importlib.import_module', 'os')
        delete = call('operator.delitem', call('builtins.vars', module), 'system')
        data = stream(delete) + stream(call('os.system', 'unused'))
        self.assertEqual(scan(data), [])
        data = b'\x80\x04' + emit(global_('os.system')) + b'q\x000' + emit(delete) + b'0h\x00' + emit(('unused',)) + b'R.'
        self.assertTrue(scan(data))

    def test_vars_is_a_dictionary_and_new_modules_are_empty(self):
        partial = call('functools.partial', global_('builtins.str'))
        namespace = call('builtins.vars', partial)
        target = Call(indirect_get(namespace, 'get'), ('func', global_('builtins.eval')))
        self.assertVerdict(Call(target, ('1+1',)), True)
        fresh = call('types.ModuleType', 'os')
        self.assertVerdict(fresh, False)
        self.assertVerdict(Call(indirect_get(fresh, 'system'), ('unused',)), False)
        no_builtins = call('builtins.vars', call('importlib.import_module', 'sys'))
        target = Call(indirect_get(no_builtins, 'get'), ('__builtins__', global_('builtins.eval')))
        self.assertVerdict(Call(target, ('1+1',)), True)

    def test_thread_state_retargeting_uses_private_fields(self):
        initial = (global_('threading.Thread'), (), {'target': global_('builtins.str'), 'args': ('1+1',)}, None)
        prefix = b'\x80\x04' + emit(call('functools.partial', global_('threading.Thread'))) + emit(initial) + b'b)R'
        for field, expected in (('_target', True), ('target', False)):
            data = prefix + emit({field: global_('builtins.eval')}) + b'bq\x000' + emit(global_('builtins.getattr')) + b'(h\x00' + emit('start') + b'tR)R.'
            self.assertEqual(bool(scan(data)), expected)

    def test_module_getattr_hooks_execute_on_missing_attributes(self):
        for getter, expected in ((global_('builtins.str'), False), (global_('os.system'), True)):
            module = call('types.ModuleType', 'model_module')
            data = b'\x80\x04' + emit(global_('builtins.getattr')) + b'(' + emit(module) + emit({'__getattr__': getter}) + b'b' + emit('unused') + b'tR.'
            self.assertEqual(bool(scan(data)), expected)

    def test_dictionary_fromkeys_consumes_a_lazy_input(self):
        source = call('builtins.map', global_('builtins.eval'), ['1+1'])
        target = indirect_get({}, 'fromkeys')
        self.assertVerdict(Call(target, (source,)), True)

    def test_special_method_callbacks_are_inert_until_used(self):
        for method, consumer in (('__bool__', 'builtins.bool'), ('__int__', 'builtins.int'), ('__float__', 'builtins.float'), ('__str__', 'builtins.str'), ('__hash__', 'builtins.hash')):
            cls = call('builtins.type', 'Value', (global_('builtins.object'),), {method: call('functools.partial', global_('subprocess.Popen'), 'unused')})
            obj = Call(cls)
            self.assertVerdict(obj, False)
            self.assertVerdict(call(consumer, obj), True)
        cls = call('builtins.type', 'Key', (global_('builtins.object'),), {'__hash__': call('functools.partial', global_('subprocess.Popen'), 'unused')})
        data = b'\x80\x04(' + emit(Call(cls)) + emit(1) + b'd.'
        self.assertTrue(scan(data))

    def test_dictionary_key_comparison_can_execute_a_capability(self):
        for comparator, expected in ((global_('os.system'), True), (call('functools.partial', global_('operator.eq'), 'collision'), False)):
            cls = call('builtins.type', 'Key', (global_('builtins.object'),), {'__hash__': call('functools.partial', global_('builtins.hash'), 'collision'), '__eq__': comparator})
            mapping = b'(' + emit(Call(cls)) + emit('value') + b'd'
            data = b'\x80\x04' + emit(global_('operator.getitem')) + b'(' + mapping + emit('collision') + b'tR.'
            self.assertEqual(bool(scan(data)), expected)

    def test_object_new_preserves_the_allocated_class(self):
        obj = Call(Global('builtins', 'object.__new__'), (global_('subprocess.Popen'),))
        self.assertVerdict(obj, False)
        self.assertVerdict(Call(indirect_get(obj, '__init__'), ('unused',)), True)


class FormatPolicy(unittest.TestCase):
    def test_compression_and_nested_containers(self):
        for obj, verdict in (({'weights': [1, 2]}, False), (call('builtins.eval', '1+1'), True)):
            raw = stream(obj)
            for compressor in (gzip.compress, bz2.compress, lzma.compress, zlib.compress):
                self.assertEqual(bool(scan(compressor(raw))), verdict)
            self.assertEqual(bool(scan(zip_bytes([('renamed.data', raw), ('card.json', b'{"info":"os.system"}')]))) , verdict)
            self.assertEqual(bool(scan(gzip.compress(zip_bytes([('model', bz2.compress(raw))])))), verdict)

    def test_gzip_multiple_members(self):
        self.assertTrue(scan(gzip.compress(stream({'a': 1})) + gzip.compress(stream(call('builtins.eval', '1+1')))))

    def test_zip_prefix_and_duplicate_last_wins(self):
        clean, bad = stream({'weights': 1}), stream(call('builtins.eval', '1+1'))
        for first, last in ((bad, clean), (clean, bad)):
            with __import__('warnings').catch_warnings():
                __import__('warnings').simplefilter('ignore')
                raw = zip_bytes([('model', first), ('model', last)], prefix=b'prefix, not a local header')
            self.assertEqual(bool(scan(raw)), last == bad)

    def test_torch_zip_and_numpy(self):
        for obj, expected in (({'weights': [1, 2]}, False), (call('builtins.eval', '1+1'), True)):
            raw = stream(obj)
            self.assertEqual(bool(scan(zip_bytes([('a/data.pkl', raw), ('a/version', b'3\n'), ('a/data/0', b'bytes')]))), expected)
            for version in (1, 2, 3):
                self.assertEqual(bool(scan(npy(raw, version=version))), expected)
            self.assertEqual(bool(scan(zip_bytes([('model.npy', npy(raw))]))), expected)
        self.assertEqual(scan(npy(stream(call('builtins.eval', '1+1')), '<f8')), [])

    def test_tf_closed_table_and_all_graphs(self):
        from modelgate.tfproto import saved_model_pb2
        from modelgate.policy import TF_OPS
        sm = saved_model_pb2.SavedModel()
        for op in ('Const', 'SaveV2', 'RestoreV2', 'MergeV2Checkpoints', 'ShardedFilename', 'ShardedFilespec'):
            n = sm.meta_graphs.add().graph_def.node.add()
            n.op, n.name = op, 'ReadFile'
            n.attr['value'].s = b'PyFunc'
        self.assertEqual(savedmodel_scan.scan_saved_model(sm.SerializeToString(), 'x'), [])
        fn = sm.meta_graphs[-1].graph_def.library.function.add()
        fn.signature.name = 'attribute_only_function'
        for op in TF_OPS:
            n = fn.node_def.add(); n.op, n.name = op, 'n_' + op
        findings = savedmodel_scan.scan_saved_model(sm.SerializeToString(), 'x')
        self.assertEqual(len(findings), len(TF_OPS))
        from google.protobuf import text_format
        self.assertEqual(len(savedmodel_scan.scan_saved_model(text_format.MessageToString(sm).encode(), 'x', True)), len(TF_OPS))

    def test_keras_nested_layers_and_named_functions(self):
        named = {'class_name': 'function', 'module': 'builtins', 'config': 'relu'}
        bytecode = {'class_name': '__lambda__', 'config': {'code': 'inert test value'}}
        for field, klass in (('function', 'Lambda'), ('output_shape', 'Lambda'), ('activation', 'Dense')):
            for value, expected in ((bytecode, True), (named, False), ('relu', False)):
                layer = {'class_name': klass, 'config': {field: value}}
                for _ in range(8): layer = {'class_name': 'TimeDistributed', 'config': {'layer': layer}}
                cfg = {'class_name': 'Model', 'config': {'layers': [layer]}}
                self.assertEqual(bool(keras_scan.scan_config(cfg, 'cfg')), expected)

    def test_scanner_never_invokes_capabilities_or_loaders(self):
        payload = stream(call('pickle.loads', stream(call('os.system', 'unused'))))
        with mock.patch('pickle.loads', side_effect=AssertionError('executed loader')), \
             mock.patch('pickle.load', side_effect=AssertionError('executed loader')), \
             mock.patch('os.system', side_effect=AssertionError('executed capability')):
            self.assertTrue(scan(payload))



    def test_large_storage_bodies_do_not_need_materialization(self):
        from pathlib import Path
        cases = [zip_bytes([('a/data.pkl', stream({'weights': 1})), ('a/version', b'3\n'), ('a/data/0', b'0' * 4096)]),
                 npy(b'0' * 4096, '<f8'),
                 b''.join([stream(TORCH_MAGIC), stream(1001), stream({}), stream({'weights': 1}), stream([])]) + b'0' * 4096]
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / 'artifact'
            for data in cases:
                artifact.write_bytes(data)
                with mock.patch('modelgate.dispatch.MAX_BYTES', 1024):
                    self.assertEqual(dispatch.scan_path(str(artifact)).verdict, 'clean')

    def test_real_numpy_pickles_are_data(self):
        import numpy as np
        objects = [np.arange(12).reshape(3, 4), np.array([os.system, str, 'eval(1)'], dtype=object),
                   np.array([('os.system', 3)], dtype=[('name', 'O'), ('value', 'i4')])]
        for obj in objects:
            for proto in range(6):
                with self.subTest(dtype=obj.dtype, proto=proto):
                    self.assertEqual(scan(pickle.dumps(obj, protocol=proto)), [])
            out = io.BytesIO()
            np.save(out, obj, allow_pickle=True)
            self.assertEqual(scan(out.getvalue()), [])


    def test_embedded_archives_remain_data_until_loaded(self):
        bad = zip_bytes([('model', stream(call('builtins.eval', '1+1')))])
        clean = zip_bytes([('model', stream({'data': 1}))])
        self.assertEqual(scan(stream({'bytes': bad})), [])
        self.assertEqual(scan(stream({'first': 1}) + stream({'bytes': bad})), [])
        self.assertEqual(scan(stream(call('builtins.eval', '1+1')) + clean), [])
        # A syntactically incomplete pickle-looking SFX prefix is still a zip.
        self.assertEqual(scan(b'\x80\x04B' + struct.pack('<I', len(clean)) + clean), [])
        self.assertTrue(scan(stream({'first': 1}) + stream(call('pickle.loads', stream(call('builtins.eval', '1+1'))))))
        # Raw tensor storage after the fifth legacy pickle is not an archive.
        legacy = b''.join([stream(TORCH_MAGIC), stream(1001), stream({}), stream({'data': 1}), stream([])])
        self.assertEqual(scan(legacy + bad), [])


    def test_savedmodel_tensor_archives_are_inert_data(self):
        from modelgate.tfproto import saved_model_pb2
        bad = zip_bytes([('model', stream(call('builtins.eval', '1+1')))])
        for graph_op, expected in (('Identity', False), ('ReadFile', True)):
            sm = saved_model_pb2.SavedModel()
            graph = sm.meta_graphs.add().graph_def
            node = graph.node.add(); node.op = 'Const'; node.name = 'bytes'
            node.attr['value'].s = bad
            node = graph.node.add(); node.op = graph_op; node.name = 'node'
            archive = zip_bytes([('model/saved_model.pb', sm.SerializeToString())])
            self.assertEqual(bool(scan(archive)), expected)

    def test_zip64_eocd_with_32_bit_entries(self):
        raw = zip_bytes([('a/data.pkl', stream(call('builtins.eval', '1+1'))), ('a/version', b'3\n')])
        eocd = len(raw) - 22
        fields = struct.unpack_from('<4s4H2IH', raw, eocd)
        record = struct.pack('<4sQ2H2I4Q', b'PK\x06\x06', 44, 45, 45, 0, 0, fields[4], fields[4], fields[5], fields[6])
        locator = struct.pack('<4sIQI', b'PK\x06\x07', 0, eocd, 1)
        self.assertTrue(scan(raw[:eocd] + record + locator + raw[eocd:]))

    def test_auxiliary_archive_text_and_gzip_padding(self):
        data = zip_bytes([('README.md', b'Model documentation, ordinary text'), ('model', stream({'data': 1}))])
        self.assertEqual(scan(data), [])
        self.assertEqual(scan(gzip.compress(stream({'weights': 1})) + b'\x00' * 32), [])
        raw = zip_bytes([('model', stream({'weights': 1}))])
        self.assertEqual(scan(b'\x1f\x8bjunk-prefix' + raw), [])



    def test_keras_metadata_is_not_a_layer(self):
        bytecode = {'class_name': '__lambda__', 'config': {'code': 'inert'}}
        data = {'class_name': 'Lambda', 'config': {'function': bytecode}}
        model = {'class_name': 'Sequential', 'config': {'layers': [{'class_name': 'Dense', 'config': {'activation': 'relu', 'notes': data}}]}, 'metadata': data}
        self.assertEqual(keras_scan.scan_config(model, 'model'), [])

    def test_cli_multiple_paths_exact_schema_and_usage(self):
        import subprocess
        from pathlib import Path
        cli = str(Path(__file__).resolve().parents[1] / 'cli.py')
        with tempfile.TemporaryDirectory() as tmp:
            paths = [str(Path(tmp) / 'clean model.bin'), str(Path(tmp) / 'unsafe model.bin')]
            Path(paths[0]).write_bytes(stream({'data': 'os.system'}))
            Path(paths[1]).write_bytes(stream(call('builtins.eval', '1+1')))
            report = str(Path(tmp) / 'report.json')
            result = subprocess.run([sys.executable, '-I', cli, 'scan', '--output', report, *paths], cwd=tmp, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads(Path(report).read_text())
            self.assertEqual(set(document), {'scanner', 'results'})
            self.assertEqual(document['scanner'], 'modelgate')
            self.assertEqual([entry['path'] for entry in document['results']], paths)
            self.assertEqual([entry['verdict'] for entry in document['results']], ['clean', 'malicious'])
            for entry in document['results']:
                self.assertEqual(set(entry), {'path', 'verdict', 'findings'})
                self.assertEqual(bool(entry['findings']), entry['verdict'] == 'malicious')
                for finding in entry['findings']:
                    self.assertEqual(set(finding), {'kind', 'location', 'detail'})
                    self.assertTrue(all(isinstance(value, str) for value in finding.values()))
            bad = subprocess.run([sys.executable, '-I', cli, 'scan'], cwd=tmp, capture_output=True, timeout=15)
            self.assertEqual(bad.returncode, 2)

    def test_cli_isolated_deployment_and_audit(self):
        import shutil
        import subprocess
        from pathlib import Path
        package = Path(__file__).resolve().parents[1]
        sample_root = package.parent / 'samples'
        with tempfile.TemporaryDirectory() as tmp:
            deployed = Path(tmp) / 'modelgate'
            shutil.copytree(package, deployed, ignore=shutil.ignore_patterns('__pycache__'))
            artifact = Path(tmp) / 'data'
            report = str(Path(tmp) / 'report.json')
            artifact.write_bytes(stream(call('pickle.loads', stream(call('builtins.eval', '1+1')))))
            runner = str(deployed / 'tests' / 'audit_runner.py')
            env = {k: os.environ[k] for k in ('PATH', 'HOME', 'TMPDIR') if k in os.environ}
            for path in [artifact] + ([sample_root / name for name in json.loads((sample_root / 'labels.json').read_text())] if sample_root.exists() else []):
                result = subprocess.run([sys.executable, '-I', runner, report, str(path)], cwd=tmp, env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                entry = json.loads(Path(report).read_text())['results'][0]
                self.assertEqual(entry['path'], str(path))
            self.assertFalse(list(deployed.rglob('*.pyc')))


if __name__ == '__main__': unittest.main()
