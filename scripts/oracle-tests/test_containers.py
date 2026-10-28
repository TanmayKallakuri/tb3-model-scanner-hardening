"""Content sniffing, archive recursion, numpy headers, caps, and the CLI contract."""
import bz2
import gzip
import io
import json
import lzma
import os
import subprocess
import sys
import tarfile
import zipfile
import zlib

import pytest

from conftest import BENIGN_P4, BINPERSID, OS_SYSTEM_P4, POP, PROTO4, SOLUTION, STOP, call, kinds, su
from modelgate import containers, dispatch
from modelgate.report import Unsupported

scan = containers.scan_bytes
KINDS = {"code-execution", "process", "file-access", "network", "unsupported"}


def make_zip(entries, leading=b""):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)
    return leading + buf.getvalue()


def make_tar(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def make_npy(descr, body, shape="(1,)"):
    header = f"{{'descr': {descr!r}, 'fortran_order': False, 'shape': {shape}, }}".encode()
    header += b" " * (63 - (len(header) + 10) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + body


# --- compression ---------------------------------------------------------------

def test_gzip_second_member_payload():
    found = scan(gzip.compress(BENIGN_P4) + gzip.compress(OS_SYSTEM_P4), "a.gz")
    assert kinds(found) == ["process"]
    assert found[0].location.startswith("a.gz|gzip:stream 1:")


def test_gzip_two_benign_members_clean():
    assert scan(gzip.compress(BENIGN_P4) + gzip.compress(BENIGN_P4), "a.gz") == []


@pytest.mark.parametrize("compress", [bz2.compress, lzma.compress, zlib.compress])
def test_other_compressors(compress):
    assert kinds(scan(compress(OS_SYSTEM_P4), "a")) == ["process"]
    assert scan(compress(BENIGN_P4), "a") == []


def test_extension_is_ignored():
    assert kinds(scan(gzip.compress(OS_SYSTEM_P4), "model.pkl")) == ["process"]
    assert kinds(scan(OS_SYSTEM_P4, "model.npz")) == ["process"]


def test_truncated_gzip_is_unsupported():
    with pytest.raises(Unsupported):
        scan(gzip.compress(BENIGN_P4)[:-6], "a.gz")


# --- zip ----------------------------------------------------------------------------

def test_zip_with_leading_bytes_and_torch_layout():
    data = make_zip([("m/data.pkl", OS_SYSTEM_P4), ("m/version", b"3"), ("m/byteorder", b"little"),
                     ("m/data/0", b"\x00" * 32), ("m/.data/serialization_id", b"0123456789")], leading=b"\x00JUNK" * 7)
    found = scan(data, "m.pt")
    assert kinds(found) == ["process"]
    assert found[0].location.startswith("m.pt/m/data.pkl:")


def test_zip_duplicate_name_last_wins():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("m/data.pkl", OS_SYSTEM_P4)
        z.writestr("m/data.pkl", BENIGN_P4)
    assert scan(buf.getvalue(), "a.zip") == []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("m/data.pkl", BENIGN_P4)
        z.writestr("m/data.pkl", OS_SYSTEM_P4)
    assert kinds(scan(buf.getvalue(), "a.zip")) == ["process"]


def test_zip_text_members_are_data():
    data = make_zip([("version", b"3"), ("notes.txt", b"2 (dup)\n"), ("data.pkl", BENIGN_P4), ("weights/0", b"\x80\x00\x00")])
    assert scan(data, "a.zip") == []


def test_npz_object_member():
    data = make_zip([("a.npy", make_npy("<f8", b"\x00" * 8)), ("b.npy", make_npy("|O", OS_SYSTEM_P4))])
    found = scan(data, "a.npz")
    assert kinds(found) == ["process"]
    assert found[0].location.startswith("a.npz/b.npy:npy-object:")


def test_keras_archive_config_is_scanned():
    lam = {"class_name": "Lambda", "config": {"name": "l", "function": {"class_name": "__lambda__", "config": {"code": "AAAA"}}}}
    cfg = {"class_name": "Functional", "config": {"layers": [lam]}}
    data = make_zip([("metadata.json", b"{}"), ("config.json", json.dumps(cfg).encode()), ("model.weights.h5", b"")])
    assert kinds(scan(data, "m.keras")) == ["code-execution"]
    cfg["config"]["layers"][0]["config"]["function"] = {"module": "builtins", "class_name": "function", "config": "relu"}
    data = make_zip([("metadata.json", b"{}"), ("config.json", json.dumps(cfg).encode())])
    assert scan(data, "m.keras") == []


def test_corrupt_zip_is_unsupported():
    data = make_zip([("a.pkl", BENIGN_P4)])
    broken = data[:20] + b"\xff" * 10 + data[30:]
    with pytest.raises(Unsupported):
        scan(broken, "a.zip")


# --- tar and nesting ----------------------------------------------------------------

def test_tar_member_payload():
    found = scan(make_tar([("w.bin", b"\x00" * 100), ("m.pkl", OS_SYSTEM_P4)]), "a.tar")
    assert kinds(found) == ["process"]
    assert found[0].location.startswith("a.tar/m.pkl:")


def test_nested_gz_tar_zip_pkl():
    inner = make_zip([("x/data.pkl", OS_SYSTEM_P4), ("x/version", b"3")], leading=b"PAD")
    data = gzip.compress(make_tar([("inner.zip", inner)]))
    found = scan(data, "a.tgz")
    assert kinds(found) == ["process"]
    assert found[0].location.startswith("a.tgz|gzip/inner.zip/x/data.pkl:")


def test_nested_benign_three_levels_clean():
    inner = make_zip([("x/data.pkl", BENIGN_P4)])
    assert scan(gzip.compress(make_tar([("inner.zip", inner)])), "a.tgz") == []


def test_depth_cap_is_unsupported():
    data = OS_SYSTEM_P4
    for _ in range(10):
        data = gzip.compress(data)
    with pytest.raises(Unsupported):
        scan(data, "deep.gz")


def test_decompression_budget_is_unsupported():
    from modelgate.report import Budget
    with pytest.raises(Unsupported):
        scan(gzip.compress(b"\x00" * 10000), "bomb.gz", 0, Budget(max_bytes=100))


def test_default_decompression_budget_is_256_mib():
    from modelgate.report import MAX_DECOMPRESSED_BYTES
    assert MAX_DECOMPRESSED_BYTES == 256 * 1024 * 1024
    assert dispatch.MAX_FILE_BYTES == 256 * 1024 * 1024


def test_file_size_cap_is_unsupported(tmp_path, monkeypatch):
    p = tmp_path / "big.pkl"
    p.write_bytes(BENIGN_P4)
    monkeypatch.setattr(dispatch, "MAX_FILE_BYTES", 4)
    result = dispatch.scan_path(str(p))
    assert kinds(result.findings) == ["unsupported"]


# --- numpy --------------------------------------------------------------------------

def test_npy_object_header_is_pickle():
    assert kinds(scan(make_npy("|O", OS_SYSTEM_P4), "o.npy")) == ["process"]
    assert scan(make_npy("|O", BENIGN_P4), "o.npy") == []


def test_npy_numeric_is_clean():
    assert scan(make_npy("<f8", b"\x00" * 8), "f.npy") == []
    assert scan(make_npy("<i4", OS_SYSTEM_P4, shape="(10,)"), "f.npy") == []


def test_npy_structured_object_field():
    body = make_npy([("a", "<f8"), ("b", "|O")], OS_SYSTEM_P4)
    assert kinds(scan(body, "s.npy")) == ["process"]


def test_npy_reads_one_stream_only():
    assert scan(make_npy("|O", BENIGN_P4 + b"\x00\x00\x00\x00"), "o.npy") == []


def test_npy_bad_header_is_unsupported():
    with pytest.raises(Unsupported):
        scan(b"\x93NUMPY\x01\x00\xff\xff{", "bad.npy")
    with pytest.raises(Unsupported):
        scan(b"\x93NUMPY\x09\x00", "bad.npy")


# --- garbage ------------------------------------------------------------------------

@pytest.mark.parametrize("data", [b"", b"\x00\x01\x02\x03", b"hello world\n", b"\x80\x04\x95\x00", b"PK\x03\x04junk",
                                  b"\x1f\x8bjunk", b"BZhjunk", b"\xfd7zXZ\x00junk", b"\x89HDF\r\n\x1a\njunk"])
def test_unrecognised_or_truncated_is_unsupported(data):
    with pytest.raises(Unsupported):
        scan(data, "g")


# --- reference files ----------------------------------------------------------------

@pytest.mark.parametrize("name", ["arrs.npz", "obj.npy", "torch_state_dict.pt", "torch_full_model.pt", "torch_ckpt.pt"])
def test_reference_containers_are_clean(refs, name):
    result = dispatch.scan_path(os.path.join(refs, "benign-files", name))
    assert result.verdict == "clean", result.findings


# --- dispatch and CLI contract --------------------------------------------------------

def test_dispatch_unsupported_is_fail_closed(tmp_path):
    p = tmp_path / "weird.bin"
    p.write_bytes(b"not a model at all")
    result = dispatch.scan_path(str(p))
    assert result.verdict == "malicious"
    assert kinds(result.findings) == ["unsupported"]
    missing = dispatch.scan_path(str(tmp_path / "missing"))
    assert kinds(missing.findings) == ["unsupported"]


def test_findings_capped_and_deduped(tmp_path):
    body = b"".join(OS_SYSTEM_P4 for _ in range(80))
    p = tmp_path / "many.pkl"
    p.write_bytes(body)
    result = dispatch.scan_path(str(p))
    assert result.verdict == "malicious"
    assert 1 <= len(result.findings) <= 50


def _run_cli(output, *paths):
    cli = os.path.join(SOLUTION, "modelgate", "cli.py")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
           "TMPDIR": os.environ.get("TMPDIR", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    return subprocess.run([sys.executable, "-I", cli, "scan", "--output", str(output), *paths],
                          cwd=str(os.path.dirname(output)), capture_output=True, text=True, env=env)


def _check_schema(doc, paths):
    assert doc["scanner"] == "modelgate"
    assert [r["path"] for r in doc["results"]] == list(paths)
    for r in doc["results"]:
        assert set(r) == {"path", "verdict", "findings"}
        assert r["verdict"] in ("malicious", "clean")
        assert (r["verdict"] == "malicious") == bool(r["findings"])
        for f in r["findings"]:
            assert set(f) == {"kind", "location", "detail"}
            assert f["kind"] in KINDS
            assert all(isinstance(f[k], str) for k in f)


def test_cli_end_to_end(tmp_path, refs):
    mal = tmp_path / "m.pkl"
    mal.write_bytes(OS_SYSTEM_P4)
    clean = tmp_path / "c.pkl"
    clean.write_bytes(BENIGN_P4)
    weird = tmp_path / "w.dat"
    weird.write_bytes(b"\x00garbage")
    sm = os.path.join(refs, "tf-files", "nested_readfile")
    out = tmp_path / "report.json"
    paths = [str(mal), str(clean), str(weird), sm, str(tmp_path / "nope")]
    proc = _run_cli(out, *paths)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    doc = json.loads(out.read_text(encoding="utf-8"))
    _check_schema(doc, paths)
    verdicts = [r["verdict"] for r in doc["results"]]
    assert verdicts == ["malicious", "clean", "malicious", "malicious", "malicious"]
    assert doc["results"][3]["findings"][0]["kind"] == "file-access"


def test_cli_usage_error_exits_2(tmp_path):
    cli = os.path.join(SOLUTION, "modelgate", "cli.py")
    proc = subprocess.run([sys.executable, "-I", cli, "scan"], capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode == 2
    assert proc.stdout == ""
    proc = subprocess.run([sys.executable, "-I", cli], capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode == 2


# --- numpy descr spellings ------------------------------------------------------------

@pytest.mark.parametrize("descr", ["object", "O", "|O", "<O8", "3O", "(2,)|O", "object_",
                                   [("a", "<f8"), ("b", "|O")], [("a", "|O", (2,))], [("a", [("b", "O")])],
                                   {"names": ["a"], "formats": ["O"]}])
def test_every_object_descr_spelling_reads_the_pickle(descr):
    assert kinds(scan(make_npy(descr, OS_SYSTEM_P4), "o.npy")) == ["process"]


@pytest.mark.parametrize("descr", ["<f8", "|u1", "<M8[ns]", "<U5", "|S3", "<c16", "|b1", "|V8",
                                   [("a", "<f8"), ("b", "<i4")], [("a", "<f8", (2, 2))], "(3,)<i4"])
def test_object_free_descrs_are_clean(descr):
    assert scan(make_npy(descr, OS_SYSTEM_P4, shape="(1,)"), "f.npy") == []


def test_unparseable_or_oversized_descr_is_unsupported():
    with pytest.raises(Unsupported):
        scan(make_npy("not a dtype", b""), "bad.npy")
    with pytest.raises(Unsupported):
        scan(make_npy("x" * 5000, b""), "bad.npy")


# --- partial findings and tar duplicates ------------------------------------------------

def test_archive_member_keeps_findings_found_before_parse_failure():
    payload = b"cos\nsystem\n(S'id'\ntR." + b"\x00\x00not a pickle"
    data = make_tar([("notes.txt", payload)])
    found = scan(data, "a.tar")
    assert kinds(found) == ["process"]
    found = scan(make_zip([("blob", payload)]), "a.zip")
    assert kinds(found) == ["process"]


def test_top_level_unsupported_keeps_partial_findings(tmp_path):
    p = tmp_path / "m.pkl"
    p.write_bytes(OS_SYSTEM_P4 + b"\x00garbage")
    result = dispatch.scan_path(str(p))
    assert result.verdict == "malicious"
    assert kinds(result.findings) == ["process", "unsupported"]


def test_tar_duplicate_member_last_wins():
    assert scan(make_tar([("m.pkl", OS_SYSTEM_P4), ("m.pkl", BENIGN_P4)]), "a.tar") == []
    assert kinds(scan(make_tar([("m.pkl", BENIGN_P4), ("m.pkl", OS_SYSTEM_P4)]), "a.tar")) == ["process"]


# --- loads that stop inside containers -------------------------------------------

def test_npy_header_written_by_python_2_is_read():
    import struct
    assert scan(make_npy("<f4", struct.pack("<4f", 1, 2, 3, 4), shape="(4L,)"), "a.npy") == []
    assert kinds(scan(make_npy("|O", OS_SYSTEM_P4, shape="(1L,)"), "a.npy")) == ["process"]
    assert scan(make_npy("|O", BENIGN_P4, shape="(1L,)"), "a.npy") == []
    for shape in ("(((", "(1L,", "'unterminated"):
        with pytest.raises(Unsupported):
            scan(make_npy("<f4", b"", shape=shape), "a.npy")


def test_archive_member_whose_load_fails_leaves_the_others():
    stops = PROTO4 + su("s") + BINPERSID + STOP
    assert kinds(scan(make_zip([("a.pkl", stops), ("b.pkl", OS_SYSTEM_P4)]), "m.zip")) == ["process"]
    assert kinds(scan(make_tar([("a.pkl", stops), ("b.pkl", OS_SYSTEM_P4)]), "m.tar")) == ["process"]
    assert scan(make_zip([("a.pkl", stops)]), "m.zip") == []


def test_torch_checkpoint_data_pkl_resolves_persistent_ids():
    payload = PROTO4 + su("storage") + BINPERSID + POP + call("os", "system", su("id")) + STOP
    assert kinds(scan(make_zip([("archive/data.pkl", payload), ("archive/version", b"3\n")]), "m.pt")) == ["process"]
    # any other pickle member is a plain pickle.load, which stops at the persistent id
    assert scan(make_zip([("archive/notes.pkl", payload)]), "m.zip") == []
