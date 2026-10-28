# ModelGate

The registry entry point and report schema are unchanged:

```sh
python3 -I /app/modelgate/cli.py scan --output /tmp/report.json /path/to/artifact
```

All implementation files and the vendored TensorFlow protobuf definitions live
in this package. No model framework is imported to load an artifact.

## Static analysis

- `pickle_scan.py` interprets pickle instructions over private symbolic values.
  It models the stack, marks, memo identity, protocol-dependent globals,
  constructors versus `__new__`, instance state, persistent IDs, frame read-ahead,
  concatenated streams, legacy PyTorch storage boundaries, and joblib arrays.
- `abstract.py` models capability calls and the ways values reach them: module
  dictionaries and aliases, attributes, partials and state changes, wrappers,
  callbacks, lazy iteration, threads, executors, and nested deserialization.
  Nested loaders use their own stream and format rules. Their payloads can be
  decoded or assembled as ordinary data, but never executed. `marshal_scan.py`
  represents code objects symbolically without creating native code objects.
- `dispatch.py` identifies formats from content, reads containers without
  extracting files, respects ZIP central directories and duplicate names, and
  distinguishes NumPy numeric buffers and PyTorch storages from pickle streams.
  HDF5 metadata and top-level ZIP entries are read without copying all weights.
- `savedmodel_scan.py` uses the closed op table in `policy.py` across every
  metagraph and function body. Node names and constant contents are data.
- `keras_scan.py` traverses nested model and layer configurations and identifies
  serialized bytecode in Lambda functions, output shapes, and activations.
  Named functions and activations remain clean.

Globals and suspicious-looking strings are not findings on their own. A known
loader error stops interpretation of unreachable instructions. Operations that
cannot be analysed, or exceed the bounded analysis budget, fail closed with an
`unsupported` finding.

The CLI disables bytecode-cache writes before importing other modules and
limits each artifact to 50 seconds. Expansion, iteration, recursion, and work
budgets also bound parser use. `parser_env.py` fills Python's cached processor
metadata from the OS-provided architecture before importing binary parsers;
this avoids h5py's otherwise implicit `uname -p` subprocess probe. Artifact
callables and modules are never imported or invoked by the symbolic machine.
Only the final JSON report is written by the CLI.

## Verification

From the package's parent directory:

```sh
python3 -B -m unittest discover -s modelgate/tests -v
python3 -B modelgate/tests/check_cli.py --count 650
```

The regression tests contain paired malicious and clean cases, including all
TensorFlow policy ops, all pickle protocols, object identity and mutation,
Python 2 mappings, nested loaders, lazy callbacks, native buffer serialization,
joblib buffers, benign NumPy/standard-library objects, and container boundaries.
Payloads are assembled or pickled as fixtures; they are never unpickled.

`check_cli.py` copies only the deployable package to a temporary location and
invokes its CLI with `-I`, a different working directory, and an environment
containing only `PATH`, `HOME`, and `TMPDIR`. It checks the JSON contract and
uses an audit hook to detect attempted processes, network activity, or writes
other than the report. Its 650-invocation timing pass cycles through 399 distinct
synthetic fixtures. A development run completed in 73.78 seconds, with a slowest
scan of 0.325 seconds. The supplied 30-artifact evaluation also passed with no
misses, false positives, or invalid reports.
