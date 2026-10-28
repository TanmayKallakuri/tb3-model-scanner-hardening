# ModelGate

Run `python3 -I /app/modelgate/cli.py scan --output REPORT ARTIFACT ...`.
The report contract is unchanged. No training frameworks or model loaders are
imported. Only this directory and installed Python packages are required.

The pickle scanner interprets opcodes on inert symbolic values. It preserves
memo aliases and mutable module/partial state, distinguishes constructors from
`__new__`, follows callable resolution and callbacks, and models iterator
consumption and nested-loader boundaries. Pure byte/text transformations are
implemented explicitly. Marshal code is parsed into descriptions, never Python
code objects or functions. Capability summaries are applied at calls; names and
strings stored as data are not evidence of execution. Operations that cannot
be resolved safely receive an `unsupported` finding.

Container dispatch uses content, handles compressed streams and archive members
in memory, and respects ZIP central-directory duplicate resolution. Torch
storage bytes, numeric NumPy arrays, and ordinary archive metadata are data.
SavedModel analysis uses the vendored protobuf definitions and the policy's
closed op table across every meta-graph and FunctionDef. Keras analysis reads
only the JSON configuration, distinguishing bytecode from named functions.

The CLI disables bytecode-cache writes before importing dependencies. A runtime
audit guard disallows process, socket and filesystem mutation operations,
allowing writes only to the requested report. The HDF5 reader avoids h5py's
incidental platform-helper subprocess. Size, nesting, work and elapsed-time
limits fail closed; the CLI has a 50-second per-artifact deadline.

Run the regression tests with:

```
python3 -B -m unittest discover -s /app/modelgate/tests -v
```

Tests build pickle bytes without loading them. They include paired malicious
and clean controls, container/graph/layer coverage, the isolated CLI/report
contract, and an independent audit hook covering parser imports and scanning.
A reproducible 650-artifact stress corpus can be generated with:

```
python3 -B /app/modelgate/tests/build_corpus.py /tmp/modelgate-corpus
python3 /app/evaluate.py --samples /tmp/modelgate-corpus
```

`evaluate.py` is a development harness; deployed scanning does not use it or
any sample labels.
