# ModelGate

The registry entry point remains:

```sh
python3 -I /app/modelgate/cli.py scan --output /tmp/report.json /path/to/artifact
```

The CLI accepts multiple paths, preserves their spelling and order in its JSON
report, and returns status 0 after writing the report (2 for usage errors).
Only this directory and installed Python packages are required.

## Static analysis

* `pickle_scan.py` interprets protocols 0–5 using a symbolic stack, memo and
  object state. Globals are references, not imports. Calls, construction and
  state restoration are distinct operations. Partial mutation, bound methods,
  Python 2 aliases, attribute/item lookup, and nested loaders propagate values.
  It reads every stream in a sequence, stops before legacy PyTorch storage,
  and skips joblib's inline numeric storage. Embedded deserialization returns
  symbolic values so a harmless inner load can still supply an outer callable.
* `dispatch.py` identifies formats by content and traverses compressed and
  archived inputs in memory. ZIP traversal follows the central directory and
  its last entry for a duplicate name. Native PyTorch storage is not interpreted
  as pickle code. No files are extracted.
* `numpy_scan.py` parses literal NPY headers and inspects pickle payloads only
  for object dtypes. Numerical array bytes remain data.
* `savedmodel_scan.py` uses the vendored protobuf definitions, without importing
  TensorFlow. It checks exact op names in every MetaGraph and FunctionDef,
  including function bodies referenced only by node attributes.
* `keras_scan.py` reads HDF5's `model_config` attribute or a Keras ZIP's JSON
  configuration and traverses nested models and wrapper layers. It distinguishes
  serialized Lambda bytecode from named function references.

The scanners never use `pickle.load`, `numpy.load`, a framework model loader,
`eval`, `exec`, or a callable imported from an artifact. HDF5 is opened read-only;
its dependency's optional processor subprocess probe is replaced with kernel
metadata. Bytecode cache writes are disabled before package imports. Recursive
parsers share work, expansion and time limits; the CLI also enforces a 50-second
per-artifact deadline. Parsing failures produce `unsupported` findings rather
than silently returning a clean verdict.

## Regression checks

From the parent directory:

```sh
python3 -B -m unittest discover -s modelgate/tests -v
python3 -B modelgate/tests/benchmark.py
```

The tests construct pickle opcodes as data and never unpickle their payloads.
They exercise active and inert forms of capabilities, partial aliasing and
retargeting, loader defaults, protocol variants, containers, object arrays,
Keras structures and the complete TensorFlow policy table. CLI checks copy only
this package into a fresh deployment, use isolated Python with a minimal
environment, and audit for extra writes, process creation and networking.
The benchmark generates 300 policy-derived artifacts and starts the scanner
once per artifact.
