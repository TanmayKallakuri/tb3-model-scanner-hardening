ModelGate performs static analysis and preserves the registry CLI and report
schema. All runtime code and data are contained in this directory.

* `dispatch.py` recognizes formats by content and reads containers in memory.
  ZIP entries follow the central directory and its last-name-wins lookup.
  Embedded loaders use their own format and stream rules.
* `pickle_scan.py` implements the pickle stack, memo, streams, framing, and
  construction opcodes. `capabilities.py` propagates inert values through
  operations, attributes, wrappers, callbacks, iterators, and nested loaders.
  It never resolves or calls an artifact's real Python globals. A loader error
  stops that load; independently unpacked archive members are still examined.
* `stdlib_metadata.json` records standard-library exports, aliases, types, and
  signatures. It is inert build-time metadata, not an import allowlist or a
  capability blocklist. `tests/generate_stdlib_metadata.py` regenerates it from
  the development interpreter. Call effects remain explicit in the interpreter.
* SavedModels use the vendored protobuf definitions and the closed op table.
  Keras analysis reads configuration JSON, including nested layers and wrappers.
  HDF5 is opened from memory; the platform cache avoids h5py's incidental
  `uname -p` process during import. CLI imports cannot write bytecode caches.
* Work, nesting, and expanded-data limits produce an `unsupported` finding
  rather than running arbitrary code or silently accepting unanalysed inputs.

Run regression and CLI safety tests from the package's parent directory:

```
python3 -m unittest discover -s modelgate/tests -v
```

The tests construct payload bytes without loading them. CLI tests use `-I`, a
fresh working directory, and only `PATH`, `HOME`, and `TMPDIR`. An audit hook
rejects scanner process/network operations and every file write except the
report. The external sample harness can additionally exercise supplied examples.
