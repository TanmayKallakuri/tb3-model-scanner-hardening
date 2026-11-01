# ModelGate internals

`python3 -I /app/modelgate/cli.py scan --output report.json <path>`; the report
contract is `/app/docs/report-schema.md`, the policy `/app/docs/threat-model.md`.

## How a verdict is reached

1. **Format by content** (`formats.analyze_blob`, `containers.sniff`): gzip
   (all members), bz2, xz, zlib, lzma-alone, zip (central directory decides,
   last same-name entry wins, bytes before the archive tolerated), tar, `.npy`,
   HDF5, otherwise pickle stream(s). Containers recurse (depth limit 8).
   A zip with `<prefix>/data.pkl` and a local header at offset 0 is a torch
   checkpoint (only `data.pkl` is unpickled, with torch's persistent ids); a zip
   with `config.json` + `metadata.json`/`model.weights.h5` is a `.keras` archive;
   a zip of `.npy` members is an `.npz`; any other zip/tar is an archive whose
   members are loaded by their own content. A directory (or archive directory)
   with `saved_model.pb`/`.pbtxt` is a SavedModel (`.pb` wins).
2. **Pickles** are executed by `pickle_vm.Machine`, a re-implementation of the
   C unpickler (marks/fences, memo, `PROTO`-dependent `find_class` incl.
   `fix_imports` and dotted names, `PERSID`, framing, error behaviour), over
   symbolic values. Raw files: `pickle.load` until EOF; legacy torch files
   (first pickle = magic): exactly five pickles. Where the real loader raises,
   the stream stops — nothing after it runs.
3. **Calls** go through `interp.Interp`: globals resolve to real objects of
   side-effect-free standard-library modules (never anything from the
   artifact; third-party modules are symbolic), BUILD/`setattr`/`vars()`
   rebinding, `sys.modules`, getattr/attrgetter/methodcaller/partial/map/
   reduce/threads/executors ... are modelled in `handlers.py`; a call that
   reaches a callable listed in `kb.CAPS` (after the argument checks that make
   the real call raise first) is a finding. Nested loaders (`nested.py`) are
   judged by what they would load, after decoding/assembly of the embedded data.
4. **Keras**: Lambda `function`/`output_shape` bytecode and `activation`
   `__lambda__` dicts, anywhere layers nest (`keras_scan`, `h5lite`).
   **TensorFlow**: the closed op table over every meta graph, graph node and
   library function node (`savedmodel_scan`).

## Deliberate judgement calls

* Unknown top-level modules that are neither standard library nor a known
  ML/data package (`kb.KNOWN_THIRD_PARTY`) fail to import, as they would.
* Modules in `sys.modules` lookups are assumed loaded; `nt` is `os` (policy).
* `logging.config.dictConfig`, `timeit.*`, `compile` are findings whenever
  called (seed list: they evaluate/compile source).
* Directory listing / `stat` / `exists` are not file access; `open` for
  reading is.
* Fail closed (`unsupported`) only when the scan cannot finish (time budget,
  nesting) or the emulator itself failed and nothing else was found.

## Safety of the scanner itself

No subprocesses, no network, no writes except the report (bytecode caching is
off). Real functions are only evaluated for an allowlist of pure helpers on
plain data with size guards (`interp.PURE*`, `size_ok`). HDF5 attributes are
read by the bounded `h5lite` parser; libhdf5 is only used for layouts it does
not handle. A provisional fail-closed report is written first and a C-level
watchdog (`faulthandler`) ends the process before the 60 s limit if a native
call never returns.

## Extending

Add capability callables to `kb.CAPS` by canonical name (`module.qualname`
with module aliases normalised, see `interp.canonical`); add call semantics
(callers, indirection) to `handlers.py` with `@h("canonical.name")`.
