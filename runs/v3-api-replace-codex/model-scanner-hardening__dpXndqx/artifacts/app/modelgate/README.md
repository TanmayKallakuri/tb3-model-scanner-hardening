ModelGate performs static analysis only. Its entry point remains:

```
python3 -I /app/modelgate/cli.py scan --output report.json ARTIFACT [ARTIFACT ...]
```

`dispatch.py` identifies content and walks archives without extracting files.
It follows the zip central directory (including duplicate-name precedence),
handles concatenated compressed members, and distinguishes checkpoint payloads
from storage data. Large checkpoint storage bodies and numeric NPY data are
skipped; only their relevant metadata is read. All supporting code and data are
inside this package.

`pickle_scan.py` decodes protocols 0–5 with pickletools and implements the
unpickler's stack, memo, stream, persistent-ID, construction and state rules.
`symbolic.py` tracks inert references, module namespaces, partials, wrappers,
iterators, memory streams and nested loaders. A reference is classified when it
is called, not when its name appears in data. Known load failures stop the
corresponding execution path. Unmodelled calls and exhausted analysis budgets
produce the required `unsupported` finding.

`stdlib.json` is a frozen catalog of standard-library attributes, aliases and
signatures. It contains no artifact labels or sample-specific rules. Rebuild it
with `python3 modelgate/tools/build_stdlib_catalog.py` when updating the scanner's
Python runtime. No module named by an artifact is imported during a scan.

SavedModels are parsed with the vendored protobuf schema, without TensorFlow.
All meta-graphs and function definitions use the policy's exact closed op table.
Keras configurations are read as JSON; layer bytecode is identified without
Keras or a bytecode loader. HDF5 imports prime platform's lazy diagnostic field
using the uname syscall, preventing the library import from spawning `uname`.
Native numerical worker pools are disabled. Python bytecode-cache writes are
disabled before scanner imports.

Work is bounded by a shared operation/decompression budget and a 50-second CLI
alarm. Materialized data is limited to 256 MiB per value and 512 MiB cumulatively;
archive entries, iterator consumption and recursive calls have additional
bounds. Resource exhaustion still produces a valid report.

Run policy and isolated-deployment regressions with:

```
python3 -m unittest discover -s modelgate/tests -v
```

The tests construct opcode descriptions and inert model data. They never load
an executable artifact. An audit-hook runner checks the actual isolated CLI,
including library imports, for process creation, network access and writes
outside its report. When `/app/samples` is present, it audits every supplied
format as well. `/app/evaluate.py` provides the separate sample-label check.
