"""ModelGate: the registry's static scanner for uploaded model artifacts.

Entry point: ``python3 -I /app/modelgate/cli.py scan --output <report.json> <path>``.
The report contract is documented in /app/docs/report-schema.md and the
policy the verdicts must match in /app/docs/threat-model.md.

Layout:

* ``cli``        command line; fail-closed report, time limits
* ``dispatch``   one artifact path -> result
* ``formats``    content-based format detection, archives, per-loader semantics
* ``containers`` bounded in-memory gzip/bz2/xz/zlib/lzma/zip/tar unpacking
* ``pickle_vm``  opcode-exact re-implementation of the C unpickler
* ``interp``     symbolic model of what the unpickler's calls do
* ``handlers``   semantics of indirection helpers and callers (getattr,
                 partial, map, reduce, threads, nested loaders ...)
* ``kb``         which callables carry which capability
* ``nested``     re-entrant deserialization (pickle.loads, torch.load ...)
* ``keras_scan`` Lambda / activation bytecode in Keras configurations
* ``h5lite``     bounded reader for HDF5 root attributes
* ``savedmodel_scan`` the closed TensorFlow op table
* ``tfproto``    vendored SavedModel protobuf bindings

Nothing from an artifact is ever executed: globals are resolved against
side-effect-free standard-library modules only, and every call is modelled.
"""

__version__ = "1.0.0"
