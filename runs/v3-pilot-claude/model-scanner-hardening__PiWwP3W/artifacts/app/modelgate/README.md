# ModelGate internals

`python3 -I /app/modelgate/cli.py scan --output report.json <path>` — contract in
`/app/docs/report-schema.md`, policy in `/app/docs/threat-model.md`.

The scanner never loads, imports or calls anything an artifact names. It answers
"would the standard loader execute a capability?" by emulation:

| module | role |
|---|---|
| `cli.py` | argument parsing, no bytecode writes, always writes a report and exits 0 |
| `dispatch.py` | per-path entry point, time budget, fail-closed `unsupported` on aborts |
| `loaders.py` | content-based format detection and container semantics (gzip/bz2/xz members, zip central directory with later duplicates winning, tar, npy/npz, torch zip and legacy checkpoints, joblib streams) |
| `pickle_vm.py` | opcode-level emulator of CPython's C unpickler (marks/fence, memo, protocol-dependent `find_class`, `fix_imports`, frames, persistent ids, EXT registry) and the stream drivers |
| `pyrt.py` | the Python object model the emulator runs against: attribute resolution, calls, `__new__`/BUILD semantics, lazy iterators and their consumers, indirection gadgets (getattr, attrgetter, methodcaller, partial, lru_cache, sys.modules, vars, resolve_name, ...), pure data transforms (base64, zlib, hex, joins), nested loaders |
| `values.py` | symbolic values (modules, globals, instances, partials, lazy iterators, ...) |
| `kb.py`, `kb_data.json` | static knowledge base: canonical identity of every stdlib/numpy callable an artifact may name (`posix.system` is `os.system`, `shutil.os` is `os`), class attributes, call signatures |
| `capabilities.py` | the curated policy tables: capability-bearing callables (with argument conditions), nested loaders, modelled callables |
| `keras_scan.py`, `h5lite.py` | Keras configs (`__lambda__` bytecode); a bounded pure-Python reader for HDF5 root attributes (h5py only as a fallback) |
| `savedmodel_scan.py`, `policy.py` | SavedModel graphs against the closed TF op table |

A finding is reported only when the emulated loader *calls* a capability-bearing
callable with arguments its signature accepts; when the real loader would raise
(REDUCE on a non-tuple, dotted names below protocol 4, a persistent id without a
handler, ...) emulation stops there, as loading would.

After editing `capabilities.py`, regenerate the knowledge base:

    python3 -I /app/modelgate/devtools/build_kb.py
