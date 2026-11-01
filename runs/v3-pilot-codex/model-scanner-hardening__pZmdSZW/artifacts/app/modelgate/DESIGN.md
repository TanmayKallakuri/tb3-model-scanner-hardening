ModelGate analyzes artifact bytes without loading a model.

`dispatch.py` recognizes formats by content, decompresses and traverses
containers in memory, and honors the zip central directory (including duplicate
names and prepended data). Torch storage bytes and numeric NumPy array payloads
are data. Legacy Torch checkpoints have five pickle streams; raw pickle files
can have any number. Nested loaders use their own stream consumption rules.

`pickle_scan.py` implements a symbolic pickle machine. The stack and memo hold
literal data or scanner-owned representations of globals, modules, instances,
partials, bound methods, wrappers, and lazy iterators. Resolving a GLOBAL does
not execute it and is not a finding. Calls, object construction, BUILD state,
module mutations, and iterator consumption determine when a capability is
used. Python 2 fix_imports mappings apply below protocol 3; attribute traversal
in dotted GLOBAL names applies at protocol 4 and later. Module attributes used
for missing-name/default resolution are recorded in `module_exports.py`, so
artifact-directed imports are unnecessary.

The machine never calls a resolved artifact value. A small allowlist performs
pure computations on primitive literals. Marshal decoding produces inert
constants/code objects, which are immediately represented symbolically; Python
functions are never constructed. Statically known loader errors stop that load
without interpreting later opcodes. Analysis limits or unresolved call targets
produce an unsupported finding. State is shared across streams and nested
loads of an artifact, and is isolated between top-level path arguments.

`savedmodel_scan.py` parses the vendored protobuf schema, visits every meta
graph and library function, and checks exact op types against `policy.py`.
`keras_scan.py` reads JSON configuration only, including nested models and
wrappers, and checks the policy's serialized lambda fields. HDF5 reading uses
h5py without traversing links or datasets. Its dependency's lazy platform probe
is seeded from the kernel machine identifier to avoid launching `uname`.
Bytecode cache writes are disabled before scanner imports.

Regression checks:

    python3 -B -m unittest discover -s /app/modelgate/tests -v
    python3 /app/evaluate.py

The regression suite constructs opcode bytes without unpickling them. Its CLI
integration check runs with `-I`, a different working directory, a minimal
environment, and an audit hook that rejects process starts, network operations,
and writes other than the designated report. The production command and JSON
report contract are unchanged.
