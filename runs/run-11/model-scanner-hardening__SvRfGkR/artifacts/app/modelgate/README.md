# ModelGate

The command and JSON report contract are unchanged:

```sh
python3 -I /app/modelgate/cli.py scan --output /tmp/report.json /path/to/artifact
```

Everything required at runtime is in this directory, apart from the installed
Python standard library, protobuf and h5py. TensorFlow, PyTorch, Keras, joblib,
pandas and scikit-learn are never imported. Bytecode cache writes are disabled
before importing scanner dependencies. Archives and compressed members are read
in memory, without extraction or temporary files.

`dispatch.py` recognizes content and traverses compressed streams, ZIP central
directories (including duplicate names and prepended bytes), TAR, NumPy arrays,
PyTorch checkpoints, and Keras archives. A ZIP embedded in a pickle byte literal
remains data until a nested loader consumes it. PyTorch storage bytes and numeric
NumPy array contents are not interpreted as opcodes.

`pickle_scan.py` implements the pickle stack, memo and stream boundaries using
`pickletools` only for decoding. `pickle_vm.py` models references, imports,
attribute lookup, mutable modules and partials, constructors, state hooks,
iterators, callbacks and nested deserialization. It never imports or calls a
symbol obtained from an artifact. Small literal operations and data decoders
are evaluated using trusted implementations; marshal code records become inert
markers and are never made into Python functions. Calls that necessarily stop
loading (for example a non-tuple REDUCE argument) do not generate a capability
finding. Unresolved execution paths and resource limits fail closed. Analysis
stops once a pickle capability establishes its verdict.

`stdlib_symbols.json` and `stdlib_signatures.json` contain static metadata for
standard library aliases and argument binding. They are generated independently
of artifacts using `tests/build_symbols.py`; the generator is not a runtime
dependency and is never imported by the scanner.

TensorFlow inspection uses the vendored protobuf bindings and the policy's exact
op table across every meta graph and function definition. Keras inspection reads
only configuration data and distinguishes named functions from serialized lambda
bytecode, including nested models, wrappers, activations and output shapes.
HDF5 initialization uses syscall-derived platform information so h5py cannot
spawn the platform module's processor-detection subprocess.

Validation commands:

```sh
python3 -B /app/modelgate/tests/test_scanner.py
python3 -B /app/modelgate/tests/test_cli.py
python3 -B /app/modelgate/tests/benchmark.py --count 650
python3 -B /app/modelgate/tests/benchmark.py --count 26 --audit
```

These tests assemble independent fixtures without loading their reducers. The
benchmark invokes a fresh isolated Python process per artifact, from another
working directory with only PATH, HOME and TMPDIR. The audit wrapper rejects
process starts, network operations and writes outside the requested report. The
deployment test copies only the package and checks exact path preservation,
argument order, report shape, usage errors, and absence of extra files.
