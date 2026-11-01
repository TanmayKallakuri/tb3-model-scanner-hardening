"""Semantic regressions. Payloads are assembled as bytes and never loaded."""
import base64
import bz2
import gzip
import io
import json
import lzma
import marshal
import os
import pickle
import pickletools
import struct
import sys
import tarfile
import tempfile
import unittest
import zipfile
import zlib

sys.dont_write_bytecode = True
from modelgate import dispatch, keras_scan, savedmodel_scan
from modelgate.pickle_scan import Interpreter


class Expr:
    def __init__(self, data):
        self.data = data


def emit(value):
    if isinstance(value, Expr):
        return value.data
    if isinstance(value, tuple):
        return b"(" + b"".join(emit(v) for v in value) + b"t"
    if isinstance(value, list):
        return b"(" + b"".join(emit(v) for v in value) + b"l"
    if isinstance(value, dict):
        return b"(" + b"".join(emit(k) + emit(v) for k, v in value.items()) + b"d"
    data = pickletools.optimize(pickle.dumps(value, protocol=4))
    return b"".join(data[pos:(ops[i + 1][2] if i + 1 < len(ops) else len(data))] for i, (op, arg, pos) in enumerate(ops := list(pickletools.genops(data))) if op.name not in ("PROTO", "FRAME", "STOP"))


def G(module, name):
    return Expr(b"c" + module.encode() + b"\n" + name.encode() + b"\n")


def C(fn, *args):
    return Expr(emit(fn) + emit(tuple(args)) + b"R")


def A(obj, name):
    return C(G("builtins", "getattr"), obj, name)


def P(fn, *args, **kwargs):
    if not kwargs:
        return C(G("functools", "partial"), fn, *args)
    return B(C(G("functools", "partial"), fn), (fn, args, kwargs, {}))


def B(obj, state):
    return Expr(emit(obj) + emit(state) + b"b")


def M(index, obj):
    return Expr(emit(obj) + b"p" + str(index).encode() + b"\n")


def R(index):
    return Expr(b"g" + str(index).encode() + b"\n")


def S(*expressions):
    return Expr(b"".join(emit(v) + (b"0" if i + 1 < len(expressions) else b"") for i, v in enumerate(expressions)))


def dumps(value, protocol=4):
    return b"\x80" + bytes([protocol]) + emit(value) + b"."


def npy(payload, descr="|O", shape=(1,)):
    header = repr({"descr": descr, "fortran_order": False, "shape": shape}).encode() + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload


def zip_bytes(entries, prefix=b""):
    out = io.BytesIO()
    out.write(prefix)
    with zipfile.ZipFile(out, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return out.getvalue()


class PickleTests(unittest.TestCase):
    def check(self, expr, expected=False, protocol=4, kind=None, raw=False):
        vm = Interpreter()
        vm.load(expr if raw else dumps(expr, protocol), "test")
        self.assertEqual(bool(vm.findings), expected, vm.findings)
        if kind:
            self.assertEqual(vm.findings[0].kind, kind, vm.findings)
        return vm

    def test_data_not_execution(self):
        for expr in ["eval( os.system subprocess exec(", {"cmd": "os.system"}, G("os", "system"), [G("builtins", "eval")], P(G("os", "system"), "command")]:
            with self.subTest(expr=expr):
                self.check(expr)

    def test_direct_capabilities(self):
        for mod, fn, args, kind in [
            ("os", "system", ("command",), "process"), ("posix", "system", ("command",), "process"),
            ("nt", "system", ("command",), "process"), ("builtins", "eval", ("1",), "code-execution"),
            ("builtins", "compile", ("1", "x", "eval"), "code-execution"),
            ("builtins", "open", ("/outside",), "file-access"), ("io", "open", ("/outside",), "file-access"),
            ("os", "execve", ("/bin/true", [], {}), "process"),
            ("socket", "create_connection", (("localhost", 1),), "network"),
            ("urllib.request", "urlopen", ("http://example.invalid",), "network"),
        ]:
            with self.subTest(fn=fn):
                self.check(C(G(mod, fn), *args), True, kind=kind)

    def test_memo(self):
        self.check(S(M(999, G("os", "system")), C(R(999), "command")), True, kind="process")
        self.check(S(M(2, "os"), M(3, "system"), C(Expr(emit(R(2)) + emit(R(3)) + b"\x93"), "command")), True, kind="process")

    def test_global_protocol(self):
        self.check(C(G("shutil", "os.system"), "command"), True, kind="process")
        self.check(G("shutil", "os.system"))
        self.check(C(G("shutil", "os.system"), "command"), protocol=2)
        self.check(C(G("bdb", "Bdb.run"), C(G("bdb", "Bdb")), "x"), True, kind="code-execution")
        self.check(C(G("bdb", "Bdb.run"), C(G("bdb", "Bdb")), "x"), protocol=2)

    def test_python2_aliases(self):
        for mod, name, args, kind in [("__builtin__", "eval", ("1",), "code-execution"), ("commands", "getoutput", ("command",), "process"), ("urllib2", "urlopen", ("http://invalid",), "network"), ("httplib", "HTTPConnection.request", (C(G("http.client", "HTTPConnection"), "host"), "GET", "/"), "network")]:
            if "." not in name:
                self.check(C(G(mod, name), *args), True, protocol=2, kind=kind)
        self.check(C(A(C(G("httplib", "HTTPConnection"), "host"), "request"), "GET", "/"), True, protocol=2, kind="network")

    def test_indirection(self):
        imports = [C(G("builtins", "__import__"), "os"), C(G("importlib", "import_module"), "os"), G("shutil", "os")]
        for module in imports:
            self.check(module)
            self.check(C(A(module, "system"), "command"), True, kind="process")
            self.check(C(C(G("operator", "attrgetter"), "system"), module))
            self.check(C(C(C(G("operator", "attrgetter"), "system"), module), "command"), True, kind="process")
            self.check(C(C(G("operator", "methodcaller"), "system", "command"), module), True, kind="process")
        self.check(C(C(G("pkgutil", "resolve_name"), "os:system"), "command"), True, kind="process")
        self.check(C(C(G("pkgutil", "resolve_name"), "shutil.os.system"), "command"), True, kind="process")
        self.check(C(C(G("operator", "getitem"), C(G("builtins", "vars"), imports[0]), "system"), "command"), True, kind="process")
        self.check(C(A(C(G("operator", "getitem"), G("sys", "modules"), "os"), "system"), "command"), True, kind="process")
        self.check(C(G("operator", "call"), G("os", "system"), "command"), True, kind="process")

    def test_partial_state(self):
        p = B(C(G("functools", "partial"), G("builtins", "str")), (G("os", "system"), ("command",), {}, {}))
        self.check(p)
        self.check(C(p), True, kind="process")
        self.check(S(M(0, C(G("functools", "partial"), G("builtins", "str"))), B(R(0), (G("os", "system"), ("command",), {}, {})), C(R(0))), True, kind="process")
        self.check(C(C(G("copy", "copy"), P(G("os", "system"), "command"))), True, kind="process")

    def test_build_module(self):
        module = C(G("importlib", "import_module"), "builtins")
        setup = B(module, {"harmless": G("os", "system")})
        self.check(setup)
        self.check(S(setup, C(G("builtins", "harmless"), "command")), True, kind="process")
        self.check(dumps(setup) + dumps(C(G("builtins", "harmless"), "command")), True, raw=True, kind="process")

    def test_newobj_inst_obj(self):
        for cls in [G("subprocess", "Popen"), G("io", "FileIO")]:
            self.check(Expr(emit(cls) + emit(("external",)) + b"\x81"))
            self.check(Expr(emit(cls) + emit(("external",)) + emit({}) + b"\x92"))
            self.check(Expr(b"(" + emit(cls) + b"o"))
        self.check(Expr(emit(G("os", "system")) + emit(("command",)) + b"\x81"))
        self.check(Expr(b"(" + emit("command") + b"ios\nsystem\n"), True, kind="process", protocol=0)
        self.check(Expr(b"(" + emit(G("os", "system")) + emit("command") + b"o"), True, kind="process", protocol=0)
        self.check(Expr(emit(G("os", "system")) + emit(["command"]) + b"R"))

    def test_no_persistent_handler(self):
        self.check(Expr(b"Pstorage\n" + emit(C(G("os", "system"), "command"))))

    def test_network_construction(self):
        for mod, cls, args in [("socket", "socket", ()), ("http.client", "HTTPConnection", ("host", 80)), ("ftplib", "FTP", ()), ("smtplib", "SMTP", ()), ("xmlrpc.client", "ServerProxy", ("http://host",))]:
            self.check(C(G(mod, cls), *args))
        for mod, cls in [("ftplib", "FTP"), ("smtplib", "SMTP")]:
            self.check(C(G(mod, cls), "host"), True, kind="network")
            self.check(C(A(C(G(mod, cls)), "connect"), "host"), True, kind="network")
        self.check(C(A(C(G("http.client", "HTTPConnection"), "host"), "request"), "GET", "/"), True, kind="network")
        self.check(C(A(C(G("socket", "socket")), "connect"), ("localhost", 1)), True, kind="network")

    def test_paths(self):
        self.check(C(G("pathlib", "Path"), "/external"))
        self.check(C(A(C(G("pathlib", "Path"), "/external"), "write_text"), "x"), True, kind="file-access")
        self.check(C(G("pathlib", "Path.write_bytes"), C(G("pathlib", "Path"), "/external"), b"x"), True, kind="file-access")
        self.check(C(G("os", "fspath"), "os.system exec("))

    def test_wrappers(self):
        fn = C(G("functools", "lru_cache"), G("os", "system"))
        self.check(fn)
        self.check(C(fn, "command"), True, kind="process")
        self.check(C(C(C(G("functools", "lru_cache"), 4), G("os", "system")), "command"), True, kind="process")
        self.check(C(C(G("copy", "copy"), G("os", "system")), "command"), True, kind="process")

    def test_lazy_callbacks(self):
        for factory, items in [(G("builtins", "map"), ["command"]), (G("builtins", "filter"), ["command"]), (G("itertools", "starmap"), [("command",)])]:
            iterator = C(factory, G("os", "system"), items)
            self.check(iterator)
            for consumer in (G("builtins", "list"), G("builtins", "tuple"), G("builtins", "next")):
                self.check(C(consumer, iterator), True, kind="process")
            self.check(C(G("builtins", "list"), C(factory, G("os", "system"), [])))
        self.check(C(G("builtins", "list"), C(G("builtins", "map"), G("builtins", "str"), [1, 2])))
        self.check(C(P(G("builtins", "sorted"), key=G("os", "system")), ["command"]), True, kind="process")
        for fn in ("min", "max"):
            self.check(C(P(G("builtins", fn), key=G("os", "system")), ["command"]), True, kind="process")
        self.check(C(G("functools", "reduce"), G("builtins", "eval"), ["1", {}]), True, kind="code-execution")
        self.check(C(G("functools", "reduce"), G("os", "system"), ["command"]))

    def test_threads(self):
        for target, expected in [(G("os", "system"), True), (G("builtins", "str"), False)]:
            thread = C(P(G("threading", "Thread"), target=target, args=("command",)))
            self.check(thread)
            self.check(C(A(thread, "start")), expected, kind="process" if expected else None)
            executor = C(G("concurrent.futures", "ThreadPoolExecutor"))
            self.check(C(A(executor, "submit"), target, "command"), expected, kind="process" if expected else None)
        proc = C(G("multiprocessing", "Process"))
        self.check(proc)
        self.check(C(A(proc, "start")), True, kind="process")

    def test_nested_loaders(self):
        bad = dumps(C(G("os", "system"), "command"))
        clean = dumps({"dangerous data": G("os", "system")})
        for name in ("loads",):
            self.check(C(G("pickle", name), clean))
            self.check(C(G("pickle", name), bad), True, kind="process")
            self.check(C(G("pickle", name), clean + bad))
        self.check(C(G("pickle", "load"), C(G("io", "BytesIO"), clean + bad)))
        self.check(C(G("pickle", "load"), C(G("io", "BytesIO"), bad)), True, kind="process")
        self.check(C(G("pickle", "load"), "a path is not a file object"))
        self.check(C(C(G("pickle", "loads"), dumps(G("os", "system"))), "command"), True, kind="process")
        for loader in [G("torch", "load"), G("joblib", "load"), G("numpy", "load"), G("pandas", "read_pickle")]:
            self.check(C(loader, "/external"), True, kind="file-access")
        for convert, encoded in [
            (G("base64", "b64decode"), base64.b64encode(bad)),
            (G("zlib", "decompress"), zlib.compress(bad)),
            (G("builtins", "bytes.fromhex"), bad.hex()),
        ]:
            self.check(C(G("pickle", "loads"), C(convert, encoded)), True, kind="process")
            safe = base64.b64encode(clean) if convert.data == G("base64", "b64decode").data else zlib.compress(clean) if convert.data == G("zlib", "decompress").data else clean.hex()
            self.check(C(G("pickle", "loads"), C(convert, safe)))
        self.check(C(G("pickle", "loads"), C(G("operator", "add"), bad[:5], bad[5:])), True, kind="process")
        self.check(C(G("pickle", "loads"), C(G("builtins", "bytearray"), bad)), True, kind="process")

    def test_numpy_nested(self):
        bad = npy(dumps(C(G("os", "system"), "command")))
        for allow in [False, True]:
            self.check(C(P(G("numpy", "load"), allow_pickle=allow), C(G("io", "BytesIO"), bad)), allow, kind="process" if allow else None)
            self.check(C(P(G("numpy.lib.format", "read_array"), allow_pickle=allow), C(G("io", "BytesIO"), bad)), allow, kind="process" if allow else None)

    def test_multistream(self):
        bad = dumps(C(G("os", "system"), "command"))
        self.check(dumps([1, 2]) + bad, True, raw=True, kind="process")
        legacy = b"".join(pickle.dumps(v, 2) for v in [119547037146038801333356, 1001, {}]) + bad + pickle.dumps([], 2) + b"storage bytes"
        self.check(legacy, True, raw=True, kind="process")
        legacy = b"".join(pickle.dumps(v, 2) for v in [119547037146038801333356, 1001, {}, {}, []]) + bad
        self.check(legacy, raw=True)

    def test_serialized_bytecode(self):
        code = marshal.dumps(compile("1 + 2", "fixture", "eval"))
        self.check(C(G("marshal", "loads"), code))
        self.check(C(G("types", "FunctionType"), C(G("marshal", "loads"), code), {}), True, kind="code-execution")


class FormatTests(unittest.TestCase):
    def scan(self, data, expected=False, name="artifact", kind=None):
        vm = Interpreter()
        dispatch.scan_bytes(data, name, vm=vm)
        self.assertEqual(bool(vm.findings), expected, vm.findings)
        if kind:
            self.assertEqual(vm.findings[0].kind, kind)

    def test_containers(self):
        clean = dumps({"x": "os.system"})
        bad = dumps(C(G("os", "system"), "command"))
        for compress in (gzip.compress, bz2.compress, lzma.compress, zlib.compress):
            self.scan(compress(clean))
            self.scan(compress(bad), True, kind="process")
        self.scan(gzip.compress(clean) + gzip.compress(bad), True, kind="process")
        for payload, expected in [(clean, False), (bad, True)]:
            self.scan(zip_bytes([("model.xyz", payload), ("README", b"This is metadata")], b"prefix"), expected, kind="process" if expected else None)
            out = io.BytesIO()
            with tarfile.open(fileobj=out, mode="w") as archive:
                info = tarfile.TarInfo("model.xyz")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            self.scan(gzip.compress(out.getvalue()), expected, kind="process" if expected else None)
        self.scan(zip_bytes([("data.pkl", bad), ("data.pkl", clean)]))
        self.scan(zip_bytes([("data.pkl", clean), ("data.pkl", bad)]), True, kind="process")

    def test_numpy(self):
        bad = dumps(C(G("os", "system"), "command"))
        self.scan(npy(bad), True, kind="process")
        self.scan(npy(bad, "<f8"))
        self.scan(npy(bad, [("x", "<f8"), ("y", "|O")]), True, kind="process")
        self.scan(zip_bytes([("model.npy", npy(bad))]), True, kind="process")

    def test_keras(self):
        code = {"class_name": "__lambda__", "config": {"code": "base64"}}
        for field in ("function", "output_shape"):
            config = {"class_name": "Sequential", "config": {"layers": [{"class_name": "Lambda", "config": {field: code}}]}}
            self.assertTrue(keras_scan.scan_config(config, "config"))
        config = {"class_name": "Sequential", "config": {"layers": [{"class_name": "TimeDistributed", "config": {"layer": {"class_name": "Sequential", "config": {"layers": [{"class_name": "Dense", "config": {"activation": code}}]}}}}]}}
        self.assertTrue(keras_scan.scan_config(config, "config"))
        self.scan(zip_bytes([("config.json", json.dumps(config).encode()), ("metadata.json", b"{}")]), True, kind="code-execution")
        config = {"class_name": "Sequential", "config": {"layers": [{"class_name": "Lambda", "config": {"function": {"class_name": "function", "config": "relu"}, "name": "__lambda__"}}]}}
        self.assertFalse(keras_scan.scan_config(config, "config"))

    def test_tensorflow(self):
        from modelgate.tfproto import saved_model_pb2
        from modelgate.policy import TF_OPS
        from google.protobuf import text_format
        for op, kind in TF_OPS.items():
            model = saved_model_pb2.SavedModel()
            for _ in range(2):
                meta = model.meta_graphs.add()
            f = meta.graph_def.library.function.add()
            f.signature.name = "hidden"
            node = f.node_def.add()
            node.name = "harmless"
            node.op = op
            self.assertEqual(savedmodel_scan.scan_saved_model(model.SerializeToString(), "model")[0].kind, kind)
            self.assertEqual(savedmodel_scan.scan_saved_model(text_format.MessageToString(model).encode(), "model.pbtxt", text=True)[0].kind, kind)
        for op in ["SaveV2", "RestoreV2", "MergeV2Checkpoints", "ShardedFilename", "ShardedFilespec", "Identity", "readfile"]:
            model = saved_model_pb2.SavedModel()
            node = model.meta_graphs.add().graph_def.node.add()
            node.name = "ReadFile"
            node.op = op
            node.attr["payload"].s = b"PyFunc"
            self.assertFalse(savedmodel_scan.scan_saved_model(model.SerializeToString(), "model"))


if __name__ == "__main__":
    unittest.main()
