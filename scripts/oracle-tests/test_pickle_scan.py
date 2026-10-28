"""Symbolic pickle interpreter: every global form, memo indirection, chains,
nested loaders, multi-stream files and the false-positive boundary."""
import os
import pickle

import pytest

from conftest import (
    APPENDS, BENIGN_P4, BINPERSID, BUILD, EMPTY_DICT, EMPTY_LIST, EMPTY_TUPLE, MARK, MEMOIZE,
    NEWOBJ, NEWOBJ_EX, NONE, OS_SYSTEM_P4, POP, PROTO2, PROTO3, PROTO4, PROTO5, REDUCE, SETITEM, STOP,
    TUPLE, TUPLE1, TUPLE2, TUPLE3, args, bb, binget, binput, call, g, i1, kinds, long_binget, p4, sb, sg, su,
)
from modelgate import dispatch, pickle_scan
from modelgate.report import Unsupported

scan = pickle_scan.scan_pickle


# --- global forms and protocols ----------------------------------------------

@pytest.mark.parametrize("data", [
    b"cos\nsystem\n(S'id'\ntR.",                       # protocol 0 text GLOBAL, STRING, MARK/TUPLE
    b"cos\nsystem\n(Vid\ntR.",                         # protocol 0 UNICODE
    PROTO2 + g("os", "system") + b"U\x02id" + TUPLE1 + REDUCE + STOP,   # protocol 2 GLOBAL + SHORT_BINSTRING
    PROTO3 + g("os", "system") + b"X\x02\x00\x00\x00id" + TUPLE1 + REDUCE + STOP,  # BINUNICODE
    OS_SYSTEM_P4,                                       # protocol 4 STACK_GLOBAL
    PROTO5 + call("os", "system", su("id")) + STOP,     # protocol 5
    PROTO4 + b"\x95\x14\x00\x00\x00\x00\x00\x00\x00" + call("os", "system", su("id")) + STOP,  # framed
    b"(S'id'\nios\nsystem\n.",                          # INST with args
    b"(cos\nsystem\nS'id'\no.",                          # OBJ with args
])
def test_every_global_form_is_process(data):
    assert kinds(scan(data, "x")) == ["process"]


@pytest.mark.parametrize("module,name,kind", [
    ("posix", "system", "process"),
    ("nt", "system", "process"),
    ("__builtin__", "eval", "code-execution"),
    ("_io", "open", "file-access"),
    ("cPickle", "loads", "code-execution"),
])
def test_protocol0_module_aliases(module, name, kind):
    data = g(module, name) + b"(S'x'\ntR."
    found = scan(data, "x")
    assert kinds(found) == [kind]
    assert module not in found[0].detail or module == name


def test_copy_reg_reconstructor_is_benign():
    data = g("copy_reg", "_reconstructor") + MARK + g("collections", "OrderedDict") + g("__builtin__", "object") + NONE + TUPLE + REDUCE + STOP
    assert scan(data, "x") == []


def test_finding_location_and_detail_shape():
    found = scan(OS_SYSTEM_P4, "model.pkl")
    assert len(found) == 1
    assert found[0].location.startswith("model.pkl:stream 0:offset ")
    assert found[0].detail == "REDUCE os.system"


# --- memo indirection --------------------------------------------------------

def test_binget_of_memoised_global():
    body = sg("os", "system") + MEMOIZE + POP + NONE + POP + binget(0) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_long_binget_and_memoised_args():
    body = (sg("os", "system") + MEMOIZE + POP + su("id") + MEMOIZE + POP
            + long_binget(0) + long_binget(1) + TUPLE1 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_text_put_get():
    data = g("os", "system") + b"p0\n" + POP + b"g0\n" + b"(S'id'\ntR."
    assert kinds(scan(data, "x")) == ["process"]


def test_binput_binget_protocol2():
    data = PROTO2 + g("os", "system") + binput(5) + POP + binget(5) + b"U\x02id" + TUPLE1 + REDUCE + STOP
    assert kinds(scan(data, "x")) == ["process"]


def test_memoised_but_never_called_is_clean():
    body = sg("os", "system") + MEMOIZE + POP + EMPTY_LIST + MARK + binget(0) + binget(0) + APPENDS
    assert scan(p4(body), "x") == []


# --- construction without capability -----------------------------------------

@pytest.mark.parametrize("data", [
    b"(isubprocess\nPopen\n.",                                  # INST without args
    b"(csubprocess\nPopen\no.",                                  # OBJ without args
    p4(sg("subprocess", "Popen") + EMPTY_TUPLE + NEWOBJ),       # NEWOBJ
    p4(sg("subprocess", "Popen") + su("id") + TUPLE1 + NEWOBJ),  # NEWOBJ with args
    p4(sg("os", "system") + su("id") + TUPLE1 + EMPTY_DICT + NEWOBJ_EX),  # NEWOBJ_EX
    p4(sg("socket", "socket") + EMPTY_TUPLE + REDUCE),           # socket() construction
    p4(sg("ftplib", "FTP") + EMPTY_TUPLE + REDUCE),              # FTP() without host
    p4(sg("http.client", "HTTPConnection") + su("h") + TUPLE1 + REDUCE),
    p4(sg("xmlrpc.client", "ServerProxy") + su("http://h") + TUPLE1 + REDUCE),
    p4(sg("pathlib", "Path") + su("/x") + TUPLE1 + REDUCE),
    p4(sg("pathlib", "PurePosixPath") + su("/x") + TUPLE1 + REDUCE),
    p4(sg("bdb", "Bdb") + EMPTY_TUPLE + REDUCE),
    p4(sg("code", "InteractiveInterpreter") + EMPTY_TUPLE + REDUCE),
    p4(sg("io", "BytesIO") + sb(b"abc") + TUPLE1 + REDUCE),
])
def test_construction_and_dormant_classes_are_clean(data):
    assert scan(data, "x") == []


def test_dormant_globals_in_containers_are_clean():
    body = (EMPTY_DICT + su("f") + sg("os", "system") + SETITEM
            + su("g") + sg("builtins", "eval") + SETITEM
            + su("h") + EMPTY_LIST + MARK + sg("subprocess", "Popen") + sg("numpy", "log1p") + APPENDS + SETITEM)
    assert scan(p4(body), "x") == []


def test_strings_that_look_dangerous_are_data():
    obj = {"cmd": "os.system('touch /run/canary/x')", "k": "subprocess.Popen(", "e": "eval(", "n": "127.0.0.1:1023"}
    assert scan(pickle.dumps(obj, 4), "x") == []
    assert scan(pickle.dumps(obj, 0), "x") == []
    assert scan(pickle.dumps(obj, 2), "x") == []


def test_persistent_ids_are_clean():
    body = su("storage") + BINPERSID + su("k") + TUPLE2
    assert scan(p4(body), "x") == []
    assert scan(b"Pstorage0\n.", "x") == []


@pytest.mark.parametrize("data", [
    p4(call("functools", "partial", sg("operator", "add"), i1(1)) + i1(2) + TUPLE1 + REDUCE),
    p4(call("operator", "itemgetter", i1(0)) + EMPTY_LIST + MARK + i1(7) + APPENDS + TUPLE1 + REDUCE),
    p4(call("builtins", "set", EMPTY_LIST)),
    p4(call("builtins", "frozenset", EMPTY_LIST)),
    p4(call("builtins", "bytearray", sb(b"ab"))),
    p4(call("builtins", "complex", i1(1), i1(2))),
    p4(call("builtins", "range", i1(3))),
    p4(call("builtins", "slice", NONE, NONE, NONE)),
    p4(call("datetime", "datetime", sb(b"\x07\xe6\x01\x01\x00\x00\x00\x00\x00\x00"))),
    p4(call("decimal", "Decimal", su("1.5"))),
    p4(call("collections", "deque", EMPTY_LIST)),
    p4(call("_codecs", "encode", su("abc"), su("latin1"))),
    p4(call("numpy._core.multiarray", "_reconstruct", sg("numpy", "ndarray"), EMPTY_TUPLE, sb(b"b"))),
    p4(call("os", "fspath", su("/x"))),
    p4(call("posixpath", "normpath", su("/x/../y"))),
    p4(call("pydoc", "cram", su("abc"), i1(2))),
    p4(call("numpy.testing._private.utils", "assert_", i1(1))),
    p4(call("builtins", "getattr", sg("builtins", "__import__") + su("math") + TUPLE1 + REDUCE, su("sqrt")) + i1(4) + TUPLE1 + REDUCE),
])
def test_benign_callables_are_clean(data):
    assert scan(data, "x") == []


# --- BUILD and partial retargeting ---------------------------------------------

def test_partial_retargeted_by_build_then_called():
    body = (sg("functools", "partial") + sg("builtins", "len") + TUPLE1 + NEWOBJ
            + MARK + sg("os", "system") + su("id") + TUPLE1 + EMPTY_DICT + EMPTY_DICT + TUPLE
            + BUILD + EMPTY_TUPLE + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_partial_retargeted_to_benign_is_clean():
    body = (sg("functools", "partial") + sg("builtins", "len") + TUPLE1 + NEWOBJ
            + MARK + sg("builtins", "str") + su("x") + TUPLE1 + EMPTY_DICT + EMPTY_DICT + TUPLE
            + BUILD + EMPTY_TUPLE + REDUCE)
    assert scan(p4(body), "x") == []


def test_build_state_drives_later_getattr_call():
    # SimpleNamespace built by NEWOBJ, BUILD sets f=os.system, getattr(ns, 'f')('id')
    body = (sg("types", "SimpleNamespace") + EMPTY_TUPLE + NEWOBJ + MEMOIZE
            + EMPTY_DICT + su("f") + sg("os", "system") + SETITEM + BUILD
            + POP + sg("builtins", "getattr") + binget(0) + su("f") + TUPLE2 + REDUCE
            + su("id") + TUPLE1 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_build_on_plain_instance_is_clean():
    body = sg("sklearn.pipeline", "Pipeline") + EMPTY_TUPLE + NEWOBJ + EMPTY_DICT + su("steps") + EMPTY_LIST + SETITEM + BUILD
    assert scan(p4(body), "x") == []


# --- callable chains ----------------------------------------------------------

def test_getattr_import_chain():
    body = call("builtins", "getattr", call("builtins", "__import__", su("os")), su("system")) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_import_module_chain():
    body = call("builtins", "getattr", call("importlib", "import_module", su("os")), su("system")) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_attrgetter_chain():
    body = call("operator", "attrgetter", su("system")) + call("builtins", "__import__", su("os")) + TUPLE1 + REDUCE + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_attrgetter_dotted_path_chain():
    body = (call("operator", "attrgetter", su("path.join")) + call("builtins", "__import__", su("os")) + TUPLE1 + REDUCE
            + su("a") + su("b") + TUPLE2 + REDUCE)
    assert scan(p4(body), "x") == []
    body = (call("operator", "attrgetter", su("Path.write_text")) + call("builtins", "__import__", su("pathlib")) + TUPLE1 + REDUCE
            + su("/x") + su("y") + TUPLE2 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["file-access"]


def test_methodcaller_chain():
    body = call("operator", "methodcaller", su("system"), su("id")) + call("builtins", "__import__", su("os")) + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_methodcaller_connect_on_socket_instance():
    body = (call("operator", "methodcaller", su("connect"), su("127.0.0.1") + i1(1) + TUPLE2)
            + call("socket", "socket") + TUPLE1 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["network"]


def test_getattr_connect_on_http_connection_instance():
    body = call("builtins", "getattr", call("http.client", "HTTPConnection", su("h")), su("request")) + su("GET") + su("/") + TUPLE2 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["network"]


def test_ftp_with_host_connects():
    assert kinds(scan(p4(call("ftplib", "FTP", su("h"))), "x")) == ["network"]
    assert kinds(scan(p4(call("smtplib", "SMTP", su("h"), i1(25))), "x")) == ["network"]
    assert scan(p4(call("smtplib", "SMTP", su(""))), "x") == []


def test_path_write_text_via_getattr():
    body = call("builtins", "getattr", call("pathlib", "Path", su("/x")), su("write_text")) + su("y") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["file-access"]
    body = call("builtins", "getattr", call("pathlib", "Path", su("/x")), su("with_suffix")) + su(".y") + TUPLE1 + REDUCE
    assert scan(p4(body), "x") == []


def test_partial_of_dangerous_callable_called():
    body = call("functools", "partial", sg("os", "system"), su("id")) + EMPTY_TUPLE + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_partial_never_called_is_clean():
    body = call("functools", "partial", sg("os", "system"), su("id"))
    assert scan(p4(body), "x") == []


def test_dynamic_module_name_via_join():
    joined = call("operator", "methodcaller", su("join"), EMPTY_LIST + MARK + su("o") + su("s") + APPENDS) + su("") + TUPLE1 + REDUCE
    name = call("operator", "add", su("sys"), su("tem"))
    body = call("builtins", "getattr", sg("builtins", "__import__") + joined + TUPLE1 + REDUCE, name) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_dynamic_name_via_map_chr():
    chars = EMPTY_LIST + MARK + i1(111) + i1(115) + APPENDS
    modname = call("operator", "methodcaller", su("join"), call("builtins", "map", sg("builtins", "chr"), chars)) + su("") + TUPLE1 + REDUCE
    body = call("builtins", "getattr", sg("builtins", "__import__") + modname + TUPLE1 + REDUCE, su("system")) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_import_of_non_literal_name_fails_closed():
    # the module name is the result of a call the model does not follow
    body = call("builtins", "__import__", call("os", "urandom", i1(8)))
    assert kinds(scan(p4(body), "x")) == ["code-execution"]


def test_sys_modules_lookup_chain():
    body = (call("builtins", "getattr", call("operator", "getitem", sg("sys", "modules"), su("os")), su("system"))
            + su("id") + TUPLE1 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_reduce_chain_calls_function():
    body = call("functools", "reduce", sg("os", "system"), EMPTY_LIST + MARK + su("id") + su("x") + APPENDS)
    assert kinds(scan(p4(body), "x")) == ["process"]


# --- exec terminals ------------------------------------------------------------

@pytest.mark.parametrize("mod,name", [
    ("builtins", "eval"), ("builtins", "exec"), ("builtins", "compile"), ("timeit", "timeit"),
    ("runpy", "run_path"), ("runpy", "_run_code"), ("code", "interact"), ("bdb.Bdb", "run"),
    ("pdb", "run"), ("numpy.testing._private.utils", "runstring"), ("logging.config", "dictConfig"),
    ("ctypes", "CDLL"), ("importlib.machinery", "SourceFileLoader"),
])
def test_exec_terminals(mod, name):
    found = scan(p4(call(mod, name, su("x"))), "x")
    assert found and found[0].kind in ("code-execution", "file-access")


def test_marshal_loads_into_function_type():
    body = call("types", "FunctionType", call("marshal", "loads", sb(b"\xe3\x00")), EMPTY_DICT)
    assert kinds(scan(p4(body), "x")) == ["code-execution"]


def test_marshal_loads_alone_is_clean():
    assert scan(p4(call("marshal", "loads", sb(b"\xe3\x00"))), "x") == []


@pytest.mark.parametrize("mod,name,kind", [
    ("subprocess", "Popen", "process"), ("subprocess", "check_output", "process"), ("pty", "spawn", "process"),
    ("platform", "popen", "process"), ("pydoc", "pipepager", "process"), ("os", "execv", "process"),
    ("os", "posix_spawn", "process"),
    ("builtins", "open", "file-access"), ("io", "open", "file-access"), ("os", "remove", "file-access"),
    ("shutil", "rmtree", "file-access"), ("shutil", "copyfile", "file-access"), ("os", "makedirs", "file-access"),
    ("socket", "create_connection", "network"), ("urllib.request", "urlopen", "network"),
    ("ssl", "get_server_certificate", "network"), ("webbrowser", "open", "network"),
])
def test_seed_list_callables(mod, name, kind):
    assert kinds(scan(p4(call(mod, name, su("x"))), "x")) == [kind]


# --- nested loaders -------------------------------------------------------------

@pytest.mark.parametrize("mod,name", [
    ("pickle", "loads"), ("_pickle", "loads"), ("torch.storage", "_load_from_bytes"),
    ("numpy", "load"), ("joblib", "load"), ("pandas", "read_pickle"), ("dill", "loads"),
    ("cloudpickle", "loads"),
])
def test_nested_loader_on_malicious_bytes(mod, name):
    body = call(mod, name, sb(OS_SYSTEM_P4))
    found = scan(p4(body), "x")
    assert kinds(found) == ["process"]
    assert "!nested" in found[0].location


def test_nested_loader_on_benign_bytes_is_clean():
    assert scan(p4(call("pickle", "loads", sb(BENIGN_P4))), "x") == []
    assert scan(p4(call("torch", "load", call("io", "BytesIO", sb(BENIGN_P4)))), "x") == []


def test_nested_loader_via_bytesio():
    body = call("torch", "load", call("io", "BytesIO", sb(OS_SYSTEM_P4)))
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_nested_loader_via_unpickler_load():
    body = (call("builtins", "getattr", call("pickle", "Unpickler", call("io", "BytesIO", sb(OS_SYSTEM_P4))), su("load"))
            + EMPTY_TUPLE + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_nested_loader_on_base64_string():
    import base64
    encoded = base64.b64encode(OS_SYSTEM_P4).decode()
    body = call("pickle", "loads", call("base64", "b64decode", su(encoded)))
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_nested_loader_given_path_is_file_access():
    assert kinds(scan(p4(call("torch", "load", su("/tmp/x.pt"))), "x")) == ["file-access"]
    assert kinds(scan(p4(call("numpy", "load", su("/tmp/x.npy"))), "x")) == ["file-access"]


def test_nested_loader_on_unknown_data_fails_closed():
    body = call("pickle", "loads", call("os", "urandom", i1(8)))
    assert kinds(scan(p4(body), "x")) == ["code-execution"]


def test_nested_loader_depth_limit():
    data = OS_SYSTEM_P4
    for _ in range(10):
        data = p4(call("pickle", "loads", bb(data)))
    with pytest.raises(Unsupported):
        scan(data, "x")


def test_nested_loader_on_embedded_npy_and_gzip():
    import gzip
    body = call("pickle", "loads", bb(gzip.compress(OS_SYSTEM_P4)))
    assert kinds(scan(p4(body), "x")) == ["process"]


# --- streams --------------------------------------------------------------------

def test_second_stream_after_clean_stop():
    data = BENIGN_P4 + OS_SYSTEM_P4
    found = scan(data, "x")
    assert kinds(found) == ["process"]
    assert ":stream 1:" in found[0].location


def test_three_benign_streams_are_clean():
    assert scan(BENIGN_P4 * 3, "x") == []
    assert scan(pickle.dumps(1, 0) + pickle.dumps([2], 2) + BENIGN_P4, "x") == []


def test_memo_is_per_stream():
    first = p4(sg("os", "system") + MEMOIZE)
    second = p4(binget(0) + su("id") + TUPLE1 + REDUCE)
    found = scan(first + second, "x")
    # a memo miss is an Unknown callable, never a finding
    assert found == []


def test_trailing_garbage_is_unsupported():
    with pytest.raises(Unsupported):
        scan(BENIGN_P4 + b"\x00\x00garbage", "x")


def test_truncated_pickle_is_unsupported():
    with pytest.raises(Unsupported):
        scan(OS_SYSTEM_P4[:-3], "x")
    with pytest.raises(Unsupported):
        scan(b"", "x")


def _legacy_torch(payload_stream):
    magic = PROTO2 + b"\x8a\x0al\xfc\x9cF\xf9 j\xa8P\x19" + STOP
    protocol = PROTO2 + b"M\xe9\x03" + STOP
    sysinfo = pickle.dumps({"protocol_version": 1001, "little_endian": True}, 2)
    keys = pickle.dumps(["0"], 2)
    return magic + protocol + sysinfo + payload_stream + keys + b"\x08\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 8


def test_legacy_torch_layout_benign_with_trailing_raw_bytes():
    assert scan(_legacy_torch(BENIGN_P4), "x") == []


def test_legacy_torch_layout_payload_in_fourth_stream():
    found = scan(_legacy_torch(OS_SYSTEM_P4), "x")
    assert kinds(found) == ["process"]
    assert ":stream 3:" in found[0].location


# --- reference files ------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "sk_logreg.p2.pkl", "sk_logreg.p4.pkl", "sk_logreg.p5.pkl", "sk_pipeline.p2.pkl", "sk_pipeline.p4.pkl",
    "sk_pipeline.p5.pkl", "sk_rf.p2.pkl", "sk_rf.p4.pkl", "sk_rf.p5.pkl", "sk_logreg.joblib",
    "torch_legacy.pt",
])
def test_reference_pickles_are_clean(refs, name):
    path = os.path.join(refs, "benign-files", name)
    result = dispatch.scan_path(path)
    assert result.verdict == "clean", result.findings


# --- symbolic containers and tainted values --------------------------------------

def _tuple_of_system():
    # the REDUCE argument tuple is ((os.system,),): the builtin receives (os.system,)
    return sg("os", "system") + TUPLE1


def test_callable_laundered_through_list_and_getitem():
    body = (call("operator", "getitem", call("builtins", "list", _tuple_of_system()), i1(0))
            + su("id") + TUPLE1 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


@pytest.mark.parametrize("builder", ["list", "tuple", "sorted", "reversed"])
def test_callable_survives_container_builtins(builder):
    body = (call("operator", "getitem", call("builtins", builder, _tuple_of_system()), i1(0))
            + su("id") + TUPLE1 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_callable_survives_dict_values_and_next_iter():
    pairs = EMPTY_LIST + MARK + (su("f") + sg("os", "system") + TUPLE2) + APPENDS
    values = call("builtins", "getattr", call("builtins", "dict", pairs), su("values")) + EMPTY_TUPLE + REDUCE
    body = call("builtins", "next", call("builtins", "iter", values)) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_callable_survives_setitem_and_enumerate():
    lst = EMPTY_LIST + MARK + NONE + APPENDS + MEMOIZE
    body = (lst + POP + call("operator", "setitem", binget(0), i1(0), sg("os", "system")) + POP
            + call("operator", "getitem", call("operator", "getitem", call("builtins", "enumerate", binget(0)), i1(0)), i1(1))
            + su("id") + TUPLE1 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_value_lost_to_the_evaluator_is_fail_closed():
    body = call("builtins", "max", sg("os", "system"), sg("builtins", "len")) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["code-execution"]


def test_lost_value_stored_but_never_called_is_clean():
    body = call("builtins", "max", sg("os", "system"), sg("builtins", "len"))
    assert scan(p4(body), "x") == []


def test_memo_miss_call_stays_benign():
    body = binget(7) + su("id") + TUPLE1 + REDUCE
    assert scan(p4(body), "x") == []


# --- shared structure and work budget --------------------------------------------

def test_shared_substructure_is_linear():
    import time
    body = su("x") + b"".join(b"2" + TUPLE2 for _ in range(60))
    started = time.monotonic()
    assert scan(p4(call("builtins", "len", body)), "x") == []
    assert scan(p4(call("builtins", "str", body)), "x") == []
    assert time.monotonic() - started < 5


def test_work_budget_is_unsupported_not_a_hang():
    from modelgate.report import Budget
    body = b"".join(su("k") + POP for _ in range(20000))
    with pytest.raises(Unsupported):
        scan(p4(body), "x", budget=Budget(max_work=1000))


def test_deadline_is_unsupported():
    from modelgate.report import Budget
    body = b"".join(su("k") + POP for _ in range(5000))
    with pytest.raises(Unsupported):
        scan(p4(body), "x", budget=Budget(seconds=-1))


def test_budget_breach_still_reports_earlier_findings(tmp_path):
    from modelgate.report import Budget
    body = call("os", "system", su("id")) + POP + b"".join(su("k") + POP for _ in range(20000))
    with pytest.raises(Unsupported) as info:
        scan(p4(body), "x", budget=Budget(max_work=5000))
    assert kinds(info.value.findings) == ["process"]
    p = tmp_path / "slow.pkl"
    p.write_bytes(p4(body))
    result = dispatch.scan_path(str(p), seconds=-1)
    assert result.verdict == "malicious"
    assert "unsupported" in kinds(result.findings)


# --- pure evaluation caps -------------------------------------------------------

@pytest.mark.parametrize("body", [
    call("codecs", "decode", sb(b"x\x9c"), su("zlib")),
    call("codecs", "decode", sb(b"BZh"), su("bz2_codec")),
    call("struct", "pack", su("1000000000s"), sb(b"")),
    call("builtins", "getattr", su("%999999999d"), su("__mod__")) + i1(1) + TUPLE1 + REDUCE,
    call("operator", "mod", su("%999999999d"), i1(1)),
    call("builtins", "format", i1(1), su(">999999999")),
    call("operator", "mul", call("operator", "mul", su("x"), b"J\x40\x42\x0f\x00"), b"J\x40\x42\x0f\x00"),
    call("itertools", "permutations", call("builtins", "range", i1(12))),
    call("itertools", "product", call("builtins", "range", i1(200)), call("builtins", "range", i1(200))),
    call("textwrap", "fill", call("operator", "mul", su("word "), b"J\x40\x42\x0f\x00")),
    call("builtins", "round", b"J\xa0\x86\x01\x00", b"J\x00\x36\x65\xc4"),
    call("builtins", "getattr", su("x"), su("replace")) + su("x") + call("operator", "mul", su("y"), b"J\x40\x42\x0f\x00") + TUPLE2 + REDUCE,
    call("builtins", "getattr", su("\tx"), su("expandtabs")) + b"J\x80\x96\x98\x00" + TUPLE1 + REDUCE,
])
def test_pure_evaluation_caps_are_fast(body):
    import time
    started = time.monotonic()
    assert scan(p4(body), "x") == []
    assert time.monotonic() - started < 3


def test_capped_value_called_is_fail_closed():
    body = call("codecs", "decode", sb(b"x\x9c"), su("zlib")) + EMPTY_TUPLE + REDUCE
    assert kinds(scan(p4(body), "x")) == ["code-execution"]


# --- policy corrections ---------------------------------------------------------

def test_build_opener_open_is_network():
    body = call("builtins", "getattr", call("urllib.request", "build_opener"), su("open")) + su("http://h") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["network"]
    assert scan(p4(call("urllib.request", "install_opener", call("urllib.request", "build_opener"))), "x") == []


@pytest.mark.parametrize("data", [
    p4(sg("PIL.Image", "Image") + EMPTY_TUPLE + NEWOBJ + EMPTY_DICT + su("mode") + su("RGB") + SETITEM + BUILD),
    p4(sg("requests.sessions", "Session") + EMPTY_TUPLE + NEWOBJ + EMPTY_DICT + su("headers") + EMPTY_DICT + SETITEM + BUILD),
    p4(call("os", "kill", i1(1), i1(9))),
    p4(call("os", "chdir", su("/tmp"))),
    p4(call("socket", "gethostbyname", su("localhost"))),
    p4(call("socket", "getaddrinfo", su("localhost"), i1(80))),
    p4(call("webbrowser", "GenericBrowser", su("x"))),
    p4(call("redis", "Redis")),
    p4(call("gzip", "GzipFile", NONE, su("rb"), i1(9), call("io", "BytesIO", sb(b"\x1f\x8b")))),
    p4(call("zipfile", "ZipFile", call("io", "BytesIO", sb(b"PK")))),
    p4(call("tarfile", "open", NONE, su("r"), call("io", "BytesIO", sb(b"")))),
    p4(call("pickle", "dump", EMPTY_DICT, call("io", "BytesIO"))),
    p4(call("sqlite3", "Error", su("x"))),
    p4(call("shutil", "Error", su("x"))),
    p4(call("tarfile", "TarInfo", su("x"))),
    p4(call("zipfile", "BadZipFile", su("x"))),
    p4(call("pkgutil", "resolve_name", su("math"))),
    p4(call("builtins", "getattr", call("pydoc", "locate", su("math")), su("sqrt")) + i1(4) + TUPLE1 + REDUCE),
])
def test_policy_corrections_are_clean(data):
    assert scan(data, "x") == []


@pytest.mark.parametrize("data,kind", [
    (p4(call("gzip", "GzipFile", su("/etc/passwd"))), "file-access"),
    (p4(call("zipfile", "ZipFile", su("/tmp/a.zip"), su("w"))), "file-access"),
    (p4(call("tarfile", "open", su("/tmp/a.tar"))), "file-access"),
    (p4(call("bz2", "BZ2File", sb(b"/tmp/a.bz2"))), "file-access"),
    (p4(call("pickle", "dump", EMPTY_DICT, su("/tmp/x"))), "file-access"),
    (p4(call("torch", "save", EMPTY_DICT, su("/tmp/x.pt"))), "file-access"),
    (p4(call("pkgutil", "resolve_name", su("os:system")) + su("id") + TUPLE1 + REDUCE), "process"),
    (p4(call("pydoc", "locate", su("os.system")) + su("id") + TUPLE1 + REDUCE), "process"),
    (p4(call("unittest.mock", "_importer", su("subprocess.check_output")) + su("id") + TUPLE1 + REDUCE), "process"),
    (p4(call("builtins", "getattr", call("pkgutil", "resolve_name", su("os")), su("system")) + su("id") + TUPLE1 + REDUCE), "process"),
    (p4(call("builtins", "getattr", sg("zipfile", "ZipFile") + call("io", "BytesIO", sb(b"PK")) + TUPLE1 + REDUCE, su("extractall")) + EMPTY_TUPLE + REDUCE), "file-access"),
])
def test_policy_corrections_still_flag(data, kind):
    assert kinds(scan(data, "x")) == [kind]


# --- loads that stop: dotted globals, persistent ids, REDUCE arguments ----------

def test_dotted_global_resolves_attribute_by_attribute_from_protocol_4():
    assert kinds(scan(p4(call("shutil", "os.system", su("id"))), "x")) == ["process"]
    assert scan(p4(call("shutil", "os.fspath", su("/x"))), "x") == []
    assert kinds(scan(p4(call("logging", "os.popen", su("id"))), "x")) == ["process"]
    assert kinds(scan(p4(call("builtins", "eval.__call__", su("1"))), "x")) == ["code-execution"]
    assert scan(p4(call("builtins", "len.__call__", su("x"))), "x") == []
    osmod = call("importlib", "import_module", su("os"))
    body = call("builtins", "object.__getattribute__", osmod, su("system")) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]
    body = call("builtins", "object.__getattribute__", osmod, su("fspath")) + su("/x") + TUPLE1 + REDUCE
    assert scan(p4(body), "x") == []


def test_dotted_global_below_protocol_4_stops_the_load():
    assert scan(PROTO2 + g("bdb", "Bdb.run") + b"U\x02id" + TUPLE1 + REDUCE + STOP, "x") == []
    assert kinds(scan(PROTO4 + g("bdb", "Bdb.run") + su("id") + TUPLE1 + REDUCE + STOP, "x")) == ["code-execution"]
    # what ran before the failing global counts; nothing after it, in any stream
    data = (PROTO2 + g("os", "system") + b"U\x02id" + TUPLE1 + REDUCE + POP
            + g("bdb", "Bdb.run") + b"U\x02id" + TUPLE1 + REDUCE + POP
            + g("os", "popen") + b"U\x02id" + TUPLE1 + REDUCE + STOP)
    found = scan(data, "x")
    assert kinds(found) == ["process"] and len(found) == 1
    assert scan(data + OS_SYSTEM_P4, "x") == found


def test_module_reexport_is_the_module():
    body = call("builtins", "getattr", sg("shutil", "os"), su("system")) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]
    body = call("builtins", "getattr", sg("shutil", "os"), su("fspath")) + su("/x") + TUPLE1 + REDUCE
    assert scan(p4(body), "x") == []
    data = (PROTO2 + g("builtins", "getattr") + g("shutil", "os") + b"U\x06system" + TUPLE2 + REDUCE
            + b"U\x02id" + TUPLE1 + REDUCE + STOP)
    assert kinds(scan(data, "x")) == ["process"]
    # tempfile._os is os too; shutil.copy stays the file-copying function
    os_ = call("builtins", "getattr", call("builtins", "__import__", su("tempfile")), su("_os"))
    assert kinds(scan(p4(call("builtins", "getattr", os_, su("system")) + su("id") + TUPLE1 + REDUCE), "x")) == ["process"]
    assert kinds(scan(p4(call("shutil", "copy", su("/a"), su("/b"))), "x")) == ["file-access"]


def test_reexported_callables_carry_the_capability():
    assert kinds(scan(p4(call("ssl", "create_connection", su("h") + i1(1) + TUPLE2)), "x")) == ["network"]
    assert kinds(scan(p4(call("tarfile", "bltn_open", su("/x"), su("w"))), "x")) == ["file-access"]
    assert scan(p4(call("ssl", "DER_cert_to_PEM_cert", sb(b""))), "x") == []
    assert scan(p4(call("tarfile", "TarInfo", su("z"))), "x") == []


def test_persistent_id_stops_a_plain_pickle_load():
    body = EMPTY_LIST + MARK + su("storage") + BINPERSID + call("os", "system", su("id")) + APPENDS
    assert scan(p4(body), "x") == []
    assert scan(b"(Pstorage0\ncos\nsystem\n(S'id'\ntRl.", "x") == []
    # a capability that ran before the persistent id stands
    assert kinds(scan(p4(call("os", "system", su("id")) + POP + su("s") + BINPERSID), "x")) == ["process"]


def test_persistent_id_resolves_where_torch_loads():
    import io
    import zipfile
    from modelgate import containers
    payload = p4(su("storage") + BINPERSID + POP + call("os", "system", su("id")))
    assert kinds(scan(_legacy_torch(payload), "x")) == ["process"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("archive/data.pkl", payload)
        z.writestr("archive/version", b"3\n")
    assert kinds(containers.scan_bytes(buf.getvalue(), "m.pt")) == ["process"]


def test_nested_loads_reads_one_stream():
    assert scan(p4(call("pickle", "loads", bb(BENIGN_P4 + OS_SYSTEM_P4))), "x") == []
    assert kinds(scan(p4(call("pickle", "loads", bb(OS_SYSTEM_P4 + BENIGN_P4))), "x")) == ["process"]
    # torch.load reads the five streams of the legacy layout
    body = call("torch", "load", call("io", "BytesIO", bb(_legacy_torch(OS_SYSTEM_P4))))
    assert kinds(scan(p4(body), "x")) == ["process"]
    # and on plain data unpickles the first stream before rejecting it
    body = call("torch", "load", call("io", "BytesIO", bb(OS_SYSTEM_P4 + BENIGN_P4)))
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_nested_load_failure_fails_the_outer_load():
    body = call("pickle", "loads", bb(p4(su("s") + BINPERSID))) + POP + call("os", "system", su("id"))
    assert scan(p4(body), "x") == []
    body = call("torch", "load", call("io", "BytesIO", bb(BENIGN_P4))) + POP + call("os", "system", su("id"))
    assert scan(p4(body), "x") == []


def test_reduce_needs_a_tuple_of_arguments():
    # the C unpickler raises on anything but a tuple before calling
    assert scan(PROTO2 + g("os", "system") + EMPTY_LIST + MARK + b"U\x02id" + APPENDS + REDUCE + STOP, "x") == []
    assert scan(PROTO2 + g("os", "system") + b"U\x02id" + REDUCE + STOP, "x") == []
    data = (PROTO2 + g("os", "system") + g("builtins", "tuple") + EMPTY_LIST + MARK + b"U\x02id" + APPENDS
            + TUPLE1 + REDUCE + REDUCE + STOP)
    assert kinds(scan(data, "x")) == ["process"]
    # a value the model only stands in for (an attribute) may well be a tuple
    body = sg("os", "system") + call("builtins", "getattr", call("builtins", "Exception", su("id")), su("args")) + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_keyword_arguments_reach_hosts_paths_and_data():
    def with_kwargs(mod, name, key, value):
        kwargs = EMPTY_DICT + su(key) + value + SETITEM
        return (call("functools", "partial", sg(mod, name))
                + MARK + sg(mod, name) + EMPTY_TUPLE + kwargs + NONE + TUPLE + BUILD + EMPTY_TUPLE + REDUCE)
    assert kinds(scan(p4(with_kwargs("smtplib", "SMTP", "host", su("h"))), "x")) == ["network"]
    assert scan(p4(with_kwargs("smtplib", "SMTP", "timeout", i1(5))), "x") == []
    assert kinds(scan(p4(with_kwargs("torch", "load", "f", su("/x.pt"))), "x")) == ["file-access"]
    assert kinds(scan(p4(with_kwargs("pickle", "loads", "data", bb(OS_SYSTEM_P4))), "x")) == ["process"]
    assert kinds(scan(p4(with_kwargs("gzip", "open", "filename", su("/x"))), "x")) == ["file-access"]
    assert scan(p4(with_kwargs("gzip", "open", "mode", su("rb"))), "x") == []


# --- lazy iterators and the callables that consume them -------------------------

_LIST_OF_ID = EMPTY_LIST + MARK + su("id") + APPENDS
_MAP_SYSTEM = call("builtins", "map", sg("os", "system"), _LIST_OF_ID)


@pytest.mark.parametrize("body", [
    _MAP_SYSTEM,
    call("builtins", "filter", sg("os", "system"), _LIST_OF_ID),
    call("itertools", "starmap", sg("os", "system"), EMPTY_LIST + MARK + su("id") + TUPLE1 + APPENDS),
    call("builtins", "iter", _MAP_SYSTEM),
    call("itertools", "islice", _MAP_SYSTEM, i1(1)),
    call("builtins", "enumerate", _MAP_SYSTEM),
    call("builtins", "zip", _MAP_SYSTEM, EMPTY_LIST + MARK + i1(1) + APPENDS),
    call("itertools", "chain", _MAP_SYSTEM),
    call("builtins", "iter", sg("os", "system"), NONE),
    call("os", "walk", su("/tmp")),
    call("fileinput", "input", su("/x")),
    call("glob", "iglob", su("*")),
    call("asyncio", "create_subprocess_exec", su("touch"), su("/x")),
    EMPTY_DICT + su("f") + _MAP_SYSTEM + SETITEM,
])
def test_unconsumed_iterators_execute_nothing(body):
    assert scan(p4(body), "x") == []


@pytest.mark.parametrize("body,kind", [
    (call("builtins", "list", _MAP_SYSTEM), "process"),
    (call("builtins", "tuple", _MAP_SYSTEM), "process"),
    (call("builtins", "set", _MAP_SYSTEM), "process"),
    (call("builtins", "frozenset", _MAP_SYSTEM), "process"),
    (call("builtins", "sorted", _MAP_SYSTEM), "process"),
    (call("builtins", "any", _MAP_SYSTEM), "process"),
    (call("builtins", "all", _MAP_SYSTEM), "process"),
    (call("builtins", "sum", _MAP_SYSTEM), "process"),
    (call("builtins", "max", _MAP_SYSTEM), "process"),
    (call("builtins", "min", _MAP_SYSTEM), "process"),
    (call("builtins", "next", _MAP_SYSTEM), "process"),
    (call("builtins", "bytes", _MAP_SYSTEM), "process"),
    (call("builtins", "dict.fromkeys", _MAP_SYSTEM), "process"),
    (call("collections", "deque", _MAP_SYSTEM), "process"),
    (call("collections", "Counter", _MAP_SYSTEM), "process"),
    (call("builtins", "getattr", su(","), su("join")) + call("builtins", "map", sg("builtins", "str"), _MAP_SYSTEM) + TUPLE1 + REDUCE, "process"),
    (call("builtins", "list", call("itertools", "chain", _MAP_SYSTEM)), "process"),
    (call("builtins", "list", call("builtins", "zip", _MAP_SYSTEM, EMPTY_LIST + MARK + i1(1) + APPENDS)), "process"),
    (call("builtins", "list", call("builtins", "enumerate", _MAP_SYSTEM)), "process"),
    (call("builtins", "list", call("itertools", "islice", _MAP_SYSTEM, i1(1))), "process"),
    (call("builtins", "list", call("builtins", "iter", _MAP_SYSTEM)), "process"),
    (call("functools", "reduce", sg("operator", "add"), call("builtins", "map", sg("os", "system"), EMPTY_LIST + MARK + su("id") + su("id") + APPENDS)), "process"),
    (call("builtins", "list", call("builtins", "filter", sg("os", "system"), _LIST_OF_ID)), "process"),
    (call("builtins", "list", call("itertools", "starmap", sg("os", "system"), EMPTY_LIST + MARK + su("id") + TUPLE1 + APPENDS)), "process"),
    (call("builtins", "next", call("builtins", "iter", sg("os", "system"), NONE)), "process"),
    (call("builtins", "getattr", _MAP_SYSTEM, su("__next__")) + EMPTY_TUPLE + REDUCE, "process"),
    (call("builtins", "next", call("os", "walk", su("/tmp"))), "file-access"),
    (call("builtins", "list", call("fileinput", "input", su("/x"))), "file-access"),
    (call("builtins", "list", call("glob", "iglob", su("*"))), "file-access"),
])
def test_consumed_iterators_execute(body, kind):
    assert kinds(scan(p4(body), "x")) == [kind]


def test_awaited_coroutine_executes():
    body = call("asyncio", "run", call("asyncio", "create_subprocess_exec", su("touch"), su("/x")))
    assert "process" in kinds(scan(p4(body), "x"))


def test_executor_map_submits_every_call():
    pool = call("concurrent.futures", "ThreadPoolExecutor", i1(1))
    body = call("builtins", "getattr", pool, su("map")) + sg("os", "system") + _LIST_OF_ID + TUPLE2 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]
    body = call("builtins", "getattr", pool, su("map")) + sg("builtins", "len") + _LIST_OF_ID + TUPLE2 + REDUCE
    assert scan(p4(body), "x") == []


# --- keyword arguments reach key= functions ---------------------------------------

def _partial_with_key(func_mod, func_name, key_mod, key_name):
    # functools.partial(func) retargeted by BUILD to (func, (), {'key': key}, None)
    kwargs = EMPTY_DICT + su("key") + sg(key_mod, key_name) + SETITEM
    return (call("functools", "partial", sg(func_mod, func_name))
            + MARK + sg(func_mod, func_name) + EMPTY_TUPLE + kwargs + NONE + TUPLE + BUILD)


@pytest.mark.parametrize("func", ["sorted", "max", "min"])
def test_key_function_from_partial_kwargs_is_called(func):
    body = _partial_with_key("builtins", func, "os", "system") + _LIST_OF_ID + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]
    body = _partial_with_key("builtins", func, "builtins", "len") + _LIST_OF_ID + TUPLE1 + REDUCE
    assert scan(p4(body), "x") == []


def test_newobj_ex_keywords_reach_partial():
    body = (sg("functools", "partial") + sg("builtins", "sorted") + TUPLE1
            + EMPTY_DICT + su("key") + sg("os", "system") + SETITEM + NEWOBJ_EX + _LIST_OF_ID + TUPLE1 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


def test_list_sort_key_through_methodcaller_kwargs():
    def sorter(key_mod, key_name):
        kwargs = EMPTY_DICT + su("key") + sg(key_mod, key_name) + SETITEM
        caller = (call("functools", "partial", sg("operator", "methodcaller"), su("sort"))
                  + MARK + sg("operator", "methodcaller") + su("sort") + TUPLE1 + kwargs + NONE + TUPLE + BUILD
                  + EMPTY_TUPLE + REDUCE)
        return caller + _LIST_OF_ID + TUPLE1 + REDUCE
    assert kinds(scan(p4(sorter("os", "system")), "x")) == ["process"]
    assert scan(p4(sorter("builtins", "len")), "x") == []


# --- wrappers, identity returns and module objects --------------------------------

@pytest.mark.parametrize("wrap", [
    lambda f: call("copy", "copy", f),
    lambda f: call("copy", "deepcopy", f),
    lambda f: call("functools", "partial", call("copy", "copy", f)),
    lambda f: call("functools", "update_wrapper", f, sg("os", "fspath")),
    lambda f: call("functools", "wraps", sg("os", "fspath")) + f + TUPLE1 + REDUCE,
])
def test_identity_wrappers_return_the_function(wrap):
    assert kinds(scan(p4(wrap(sg("os", "system")) + su("id") + TUPLE1 + REDUCE), "x")) == ["process"]
    assert scan(p4(wrap(sg("os", "fspath")) + su("/x") + TUPLE1 + REDUCE), "x") == []


def test_wraps_returns_the_wrapper_not_the_wrapped():
    body = (call("functools", "wraps", sg("os", "system")) + sg("os", "fspath") + TUPLE1 + REDUCE
            + su("/x") + TUPLE1 + REDUCE)
    assert scan(p4(body), "x") == []


def test_reload_returns_the_module():
    mod = call("importlib", "reload", call("importlib", "import_module", su("os")))
    assert kinds(scan(p4(call("builtins", "getattr", mod, su("system")) + su("id") + TUPLE1 + REDUCE), "x")) == ["process"]
    assert scan(p4(call("builtins", "getattr", mod, su("fspath")) + su("/x") + TUPLE1 + REDUCE), "x") == []


def test_build_on_a_module_rebinds_later_globals():
    def rebind(mod, name):
        return (call("builtins", "__import__", su("posixpath"))
                + EMPTY_DICT + su("normcase") + sg(mod, name) + SETITEM + BUILD)
    body = EMPTY_LIST + MARK + rebind("os", "system") + call("posixpath", "normcase", su("id")) + APPENDS
    assert kinds(scan(p4(body), "x")) == ["process"]
    body = EMPTY_LIST + MARK + rebind("os", "fspath") + call("posixpath", "normcase", su("id")) + APPENDS
    assert scan(p4(body), "x") == []
    # os.path is posixpath, and the rebinding persists into later streams of the file
    assert kinds(scan(p4(rebind("os", "system")) + p4(call("os.path", "normcase", su("id"))), "x")) == ["process"]


def test_type_of_a_function_is_function_type():
    code = call("marshal", "loads", sb(b"\xe3\x00"))
    fn = call("builtins", "type", sg("json", "dumps")) + code + EMPTY_DICT + TUPLE2 + REDUCE
    assert kinds(scan(p4(fn + EMPTY_TUPLE + REDUCE), "x")) == ["code-execution"]
    stored = EMPTY_DICT + su("t") + call("builtins", "type", sg("json", "dumps")) + SETITEM + su("c") + code + SETITEM
    assert scan(p4(stored), "x") == []


def test_reconstructor_constructs_the_base_from_the_state():
    argv = EMPTY_LIST + MARK + su("touch") + su("/x") + APPENDS
    body = call("copyreg", "_reconstructor", sg("subprocess", "Popen"), sg("subprocess", "Popen"), argv)
    assert kinds(scan(p4(body), "x")) == ["process"]
    body = call("copyreg", "_reconstructor", sg("builtins", "list"), sg("builtins", "list"), argv)
    assert scan(p4(body), "x") == []
    # the result is an instance of cls: a later BUILD still asks about cls.__setstate__
    body = (call("copyreg", "_reconstructor", sg("functools", "partial"), sg("builtins", "object"), NONE)
            + MARK + sg("os", "system") + su("id") + TUPLE1 + EMPTY_DICT + EMPTY_DICT + TUPLE + BUILD)
    assert scan(p4(body), "x") == []
    assert scan(p4(call("builtins", "object.__new__", sg("subprocess", "Popen"), argv)), "x") == []


def test_mock_calls_its_side_effect_or_wraps():
    assert kinds(scan(p4(call("unittest.mock", "Mock", NONE, sg("os", "system")) + su("id") + TUPLE1 + REDUCE), "x")) == ["process"]
    assert scan(p4(call("unittest.mock", "Mock", NONE, sg("os", "fspath")) + su("/x") + TUPLE1 + REDUCE), "x") == []
    body = call("unittest.mock", "MagicMock", NONE, NONE, NONE, sg("os", "system")) + su("id") + TUPLE1 + REDUCE
    assert kinds(scan(p4(body), "x")) == ["process"]
    kwargs = EMPTY_DICT + su("side_effect") + sg("os", "system") + SETITEM
    body = (call("functools", "partial", sg("unittest.mock", "Mock"))
            + MARK + sg("unittest.mock", "Mock") + EMPTY_TUPLE + kwargs + NONE + TUPLE + BUILD
            + EMPTY_TUPLE + REDUCE + su("id") + TUPLE1 + REDUCE)
    assert kinds(scan(p4(body), "x")) == ["process"]


# --- constructors and methods that touch nothing ----------------------------------

def test_sysloghandler_connects_only_over_tcp_or_a_unix_socket():
    addr = su("h") + i1(1) + TUPLE2
    assert kinds(scan(p4(call("logging.handlers", "SysLogHandler", addr, i1(1), i1(1))), "x")) == ["network"]
    assert kinds(scan(p4(call("logging.handlers", "SysLogHandler", su("/dev/log"))), "x")) == ["network"]
    assert scan(p4(call("logging.handlers", "SysLogHandler", addr)), "x") == []
    assert scan(p4(call("logging.handlers", "BufferingHandler", i1(10))), "x") == []


def test_pure_path_methods_keep_the_path_class():
    pure = call("operator", "truediv", call("pathlib", "PurePosixPath", su("/d")), su("n"))
    assert scan(p4(call("builtins", "getattr", pure, su("with_suffix")) + su(".x") + TUPLE1 + REDUCE), "x") == []
    real = call("operator", "truediv", call("pathlib", "Path", su("/d")), su("n"))
    assert kinds(scan(p4(call("builtins", "getattr", real, su("write_text")) + su("x") + TUPLE1 + REDUCE), "x")) == ["file-access"]
    joined = call("builtins", "getattr", call("pathlib", "Path", su("/d")), su("joinpath")) + su("n") + TUPLE1 + REDUCE
    assert kinds(scan(p4(call("builtins", "getattr", joined, su("write_bytes")) + sb(b"") + TUPLE1 + REDUCE), "x")) == ["file-access"]


def test_shelf_is_judged_by_the_pickle_it_holds():
    def shelf(payload):
        return call("shelve", "Shelf", EMPTY_DICT + sb(b"k") + bb(payload) + SETITEM)
    assert kinds(scan(p4(call("operator", "getitem", shelf(OS_SYSTEM_P4), su("k"))), "x")) == ["process"]
    assert scan(p4(call("operator", "getitem", shelf(BENIGN_P4), su("k"))), "x") == []
    assert scan(p4(shelf(OS_SYSTEM_P4)), "x") == []
    assert kinds(scan(p4(call("builtins", "getattr", shelf(OS_SYSTEM_P4), su("values")) + EMPTY_TUPLE + REDUCE), "x")) == ["process"]
    assert kinds(scan(p4(call("shelve", "open", su("/x"))), "x")) == ["file-access"]


def test_sqlite_in_memory_database_opens_no_file():
    assert scan(p4(call("sqlite3", "connect", su(":memory:"))), "x") == []
    assert kinds(scan(p4(call("sqlite3", "connect", su("/x.db"))), "x")) == ["file-access"]


def test_hostless_client_methods_other_than_connect_do_nothing():
    ftp = call("ftplib", "FTP")
    assert scan(p4(call("builtins", "getattr", ftp, su("login")) + EMPTY_TUPLE + REDUCE), "x") == []
    assert kinds(scan(p4(call("builtins", "getattr", ftp, su("connect")) + su("h") + i1(21) + TUPLE2 + REDUCE), "x")) == ["network"]
    smtp = call("smtplib", "SMTP")
    assert scan(p4(call("builtins", "getattr", smtp, su("sendmail")) + su("a") + su("b") + su("c") + TUPLE3 + REDUCE), "x") == []
    telnet = call("telnetlib", "Telnet")
    assert kinds(scan(p4(call("builtins", "getattr", telnet, su("open")) + su("h") + TUPLE1 + REDUCE), "x")) == ["network"]


# --- Python 2 names: mapped below protocol 3, a failed import from protocol 3 on ---


def test_py2_name_mapping_applies_name_before_module():
    # protocol 0: c__builtin__ reduce is functools.reduce; reduce(os.popen, [cmd, 'r']) calls os.popen
    data = b"c__builtin__\nreduce\np0\n(cos\npopen\np1\n(lp2\nVtouch /run/canary/x\np3\naVr\np4\natp5\nRp6\n."
    assert {f.kind for f in scan(data, "p0.pkl")} == {"process"}


def test_py2_module_at_protocol_4_fails_the_load():
    # __builtin__ is not mapped at protocol 4 and does not exist: nothing after it runs
    data = PROTO4 + b"\x8c\x0b__builtin__\x8c\x04eval\x93" + b"\x8c\x05print" + TUPLE1 + REDUCE + STOP
    try:
        findings = scan(data, "p4.pkl")
    except pickle_scan.LoadFails:
        findings = []
    assert findings == []
