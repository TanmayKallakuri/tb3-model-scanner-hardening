# ModelGate

The command and JSON schema are unchanged:

```
python3 -I /app/modelgate/cli.py scan --output /tmp/report.json /path/to/artifact
```

`dispatch.py` identifies model formats by content and traverses containers in
memory. Large HDF5, checkpoint ZIPs and numeric arrays are inspected through
read-only file objects, without reading their weight data into memory. ZIP
members come from the central directory, with the final duplicate winning. Torch storage is data, legacy Torch reads five streams, and NumPy
numeric arrays are data. No archive is extracted onto the filesystem.

`pickle_scan.py` implements the pickle stack, memo, construction and state
opcodes. `symbolic.py` represents Python objects and interprets their calls and
mutations. Serialized callables remain symbols; artifact code is never imported
or executed. Imports are limited to fixed scanner and parser/data libraries.
It follows aliases, partial state, module rebinding, callbacks, lazy iteration,
object cleanup, and nested readers. Pure decoding and literal operations are bounded data
transformations. An unresolved executed callable produces `unsupported`;
a deterministic loader error stops that load without inventing an effect.

SavedModel scanning uses only the bundled protobuf definitions, examining all
meta-graphs and function definitions. Keras scanning reads configuration JSON
from HDF5 or ZIP and walks nested models and wrappers. TensorFlow and Keras
runtimes are never imported.

The CLI disables bytecode cache writes and reserves time to write a report if
analysis exceeds its time budget. HDF5 parser startup seeds `platform.uname`'s
optional processor field from its machine field, avoiding CPython's otherwise
lazy `uname -p` subprocess during h5py version discovery.

## Verification

From the directory containing `modelgate/`:

```
python3 -m unittest modelgate.tests.test_scanner -v
python3 /app/evaluate.py --scanner /app/modelgate/tests/audit_runner.py
python3 -m modelgate.tests.integration
```

The tests construct payload bytes; they never load those payloads. The audit
runner rejects subprocesses, connections, unpickling, and filesystem mutations
other than the requested report, including bytecode caches. Integration creates
560 temporary policy cases and invokes a fresh isolated audited CLI for each,
with only PATH, HOME and TMPDIR in the environment. It checks verdicts, JSON
schema, elapsed time and audit failures. Temporary fixtures are not a scanner
dependency.
