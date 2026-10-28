# ModelGate registry policy: what makes an uploaded model artifact malicious

This is the policy the registry grades scanner verdicts against. It is the
only definition that matters; the examples under `/app/samples/` illustrate it
but do not enumerate it.

## 1. Policy

An artifact is **malicious** if, and only if, loading it with its format's
standard loader, or invoking a function the artifact exports, would do any of
the following as a direct consequence, before the model is otherwise used:

- **(A) code execution**: execute attacker-controlled Python, directly or
  through indirection. Indirection includes calling a Python callable that
  evaluates or compiles source, materialising serialized bytecode into a
  function, and re-entrant deserialization: handing embedded data to another
  loader that would then do any of (A)-(D). A nested loader is judged by what
  it would load; a nested loader fed data that does none of (A)-(D) is not a
  finding.
- **(B) process**: start a process or run a shell command.
- **(C) file access**: read, write, create, delete or move a file outside the
  artifact itself.
- **(D) network**: open a network connection.

Every other artifact in scope is **clean**. A false positive and a false
negative are equally wrong.

Capabilities are the test, not names. A string that contains `os.system`, a
dictionary value `"subprocess"`, a graph node *named* `ReadFile`, or a constant
tensor holding `PyFunc` is data, not a finding. Conversely a capability reached
through a harmless-looking path (a memoised reference, a `getattr` chain, a
`functools.partial` that is retargeted later, a second pickle stream after the
first `STOP`, a function body referenced only through a node attribute) is a
finding.

Standard loaders, for the purpose of this policy:

| Format | Loader |
|---|---|
| pickle stream(s) | `pickle.load`, called repeatedly until end of file where the format is a sequence of pickles; `joblib.load` |
| PyTorch checkpoint | `torch.load(weights_only=False)` |
| NumPy | `numpy.load(allow_pickle=True)` |
| TensorFlow SavedModel | `tf.saved_model.load` with the `serve` tag set, then every exported signature is called |
| Keras HDF5 and `.keras` | `keras.models.load_model` with unsafe deserialization enabled (`safe_mode=False`, or `keras.config.enable_unsafe_deserialization()` for HDF5), then the model is called |
| compressed or archived artifact | the registry unpacks it and loads each member by that member's own format |

## 2. Formats and containers in scope

| Format | Shapes that occur | Notes |
|---|---|---|
| Raw pickle | `.pkl .pickle .bin .joblib`; protocols 0 to 5; one stream, or several concatenated streams read until end of file | protocol-0 text opcodes (`c`, `i`, `o`, `p`, `g`, `R`, `b`) are as valid as their binary forms |
| PyTorch zip checkpoint | `.pt .pth .bin .ckpt`; a zip holding `<prefix>/data.pkl`, `<prefix>/data/<n>` storages, `<prefix>/version`, `<prefix>/byteorder`; stored entries, a zip64 end-of-central-directory record with 32-bit sizes, as `torch.save` writes it | `torch.load` requires the local-header magic at offset 0 |
| PyTorch legacy checkpoint | five consecutive pickles (magic, protocol, system info, the object, storage keys) followed by raw storage bytes | `torch.load` unpickles all five streams in order; the object pickle is the fourth |
| NumPy | `.npy` with object dtype (header, then a pickle); `.npz` (a zip of `.npy` members) | |
| Compressed / archived | `.gz` (one or several members), `.bz2`, `.xz`, `.zip` (possibly with bytes before the first local file header, as `zipfile` tolerates: the central directory decides, a later entry with the same name wins), `.tar`, `.tar.gz`, `.tgz`, wrapping any in-scope artifact, nested up to three levels | content, not extension, decides the format at every level |
| TensorFlow SavedModel | a directory holding `saved_model.pb` or `saved_model.pbtxt` and `variables/` | every `meta_graphs[*]` is in scope, and within each: `graph_def.node`, every `graph_def.library.function[*].node_def`, and every function referenced through a node attribute (`If`, `StatelessIf`, `While`, `StatelessWhile`, `Case`, `PartitionedCall`, `StatefulPartitionedCall`, `MapDefun`, `For`) |
| Keras HDF5 | `.h5 .hdf5`; the root attribute `model_config` holds the model JSON | sub-models nest at any depth; wrapper layers (`TimeDistributed`, `Bidirectional`) hold a layer under `layer` |
| Keras v3 archive | `.keras`; a zip holding `config.json`, `metadata.json`, `model.weights.h5` | same JSON structure as `model_config` |

Out of scope and never submitted to the registry: ONNX, GGUF, safetensors,
TorchScript `code/*.py`, TFLite, PMML, XGBoost and LightGBM native formats,
Keras SavedModel `keras_metadata.pb`, HDF5 external links and virtual
datasets, archives with 64-bit sizes or zip64 extra fields.

## 3. Pickle semantics the policy follows

These are the facts of CPython's unpickler; the policy is defined in terms of
what the loader would actually do.

- `REDUCE` calls the callable with the arguments (a tuple: `pickle.load`'s C
  unpickler raises on anything else before calling); on a class that runs
  `__new__` and `__init__`. `INST` and `OBJ` with arguments construct fully.
  `BUILD` calls `__setstate__` on the object (for `functools.partial` that
  retargets the partial) or, when there is none, writes the state into the
  object's `__dict__`, which on a module rebinds its attributes for every
  later `GLOBAL`. These are the ways a capability executes. A dotted name in
  `GLOBAL`/`STACK_GLOBAL` (`Bdb.run`, `os.system` under module `shutil`)
  resolves attribute by attribute at protocol 4 and above and fails the load
  below that.
- `NEWOBJ` and `NEWOBJ_EX` call `__new__` only; argument-less `INST`/`OBJ`
  call `__new__` only when the target is a class, and call any other callable
  (a `functools.partial`, a function) with no arguments. `PERSID`/`BINPERSID`
  are resolved by the loader's own storage mechanism (`torch.load`) and
  execute nothing; plain `pickle.load` has no persistent-id handler and stops
  there. Importing a module executes no capability.
- A global that is only stored as data (placed in a container, bound as an
  attribute, memoised and never called) is not a finding, whatever its name.
  A global becomes a finding when the capability it carries is executed, no
  matter how it reached the executing opcode: directly, through the memo
  (`GET`, `BINGET`, `LONG_BINGET`), through `builtins.getattr`,
  `operator.attrgetter`, `operator.methodcaller`, `functools.partial`,
  `builtins.__import__` or `importlib.import_module`, or in a later stream of
  the same file. Those are examples, not a list: any route by which the
  loader ends up calling the capability counts, among them a module reached
  through `sys.modules`, `vars()` or another module's attribute
  (`shutil.os` is `os`), a name resolved by
  `pkgutil.resolve_name`, a function handed back by a wrapper such as
  `functools.lru_cache` or `copy.copy`, and a callable passed to something
  that calls it (`map`, `filter` or `itertools.starmap` once something
  consumes the iterator; `sorted`, `min` or `max` through `key`;
  `functools.reduce`; `operator.call`; a thread or executor started with
  it). A lazy iterator that is built and never consumed executes nothing.
- A nested loader reads what its own loader reads: `pickle.loads` and
  `pickle.load` read one pickle and ignore trailing bytes, `torch.load` on a
  legacy checkpoint unpickles all five streams, and embedded data that the
  pickle first decodes or assembles (`base64`, `zlib`, `bytes.fromhex`,
  concatenation, a `bytearray` literal) is judged after that step.
- Module aliases are the same module: `posix` and `nt` are `os`, `_pickle` is
  `pickle`, `_io` is `io`, and every Python 2 name the unpickler maps back
  with `fix_imports` (the `_compat_pickle` tables): `__builtin__` is
  `builtins`, `copy_reg` is `copyreg`, `commands` is `subprocess`, `urllib2`
  is `urllib.request`, `httplib` is `http.client`, and so on. Pickles written
  below protocol 3 carry those Python 2 names.

Representative capability-bearing callables (a seed list, not the policy;
anything with the same capability counts):

- (A) `builtins.eval`, `builtins.exec`, `builtins.compile`, `marshal.loads`
  with `types.FunctionType`,
  `runpy.run_path`, `runpy._run_code`, `timeit.timeit`,
  `code.InteractiveInterpreter.runsource`, `bdb.Bdb.run`, `pdb.run`,
  `numpy.testing._private.utils.runstring`, `logging.config.dictConfig`;
  nested loaders `pickle.loads`, `_pickle.loads`, `torch.load`,
  `torch.storage._load_from_bytes`, `numpy.load`,
  `numpy.lib.format.read_array`, `joblib.load`, `pandas.read_pickle`.
  `builtins.__import__`, `importlib.import_module`, `builtins.getattr`,
  `operator.attrgetter`, `operator.methodcaller` and `functools.partial`
  carry no capability of their own; they are the ways a capability-bearing
  callable is reached, and the call that reaches it is the finding.
- (B) `os.system`, `os.popen`, `os.execv`, `os.execvp`, `os.execve`,
  `os.spawnl`, `os.spawnv`, `os.posix_spawn`, `subprocess.Popen`,
  `subprocess.call`, `subprocess.run`, `subprocess.check_call`,
  `subprocess.check_output`, `pty.spawn`, `pydoc.pipepager`.
- (C) `builtins.open`, `io.open`, `os.remove`, `os.unlink`, `os.rename`,
  `os.replace`, `os.mkdir`, `os.makedirs`, `os.rmdir`, `shutil.copy`,
  `shutil.copyfile`, `shutil.move`, `shutil.rmtree`, `pathlib.Path.write_text`,
  `pathlib.Path.write_bytes`, `pathlib.Path.unlink`; a nested loader given a
  path instead of embedded data reads that path.
- (D) `socket.create_connection`, `socket.socket.connect`,
  `urllib.request.urlopen`, `http.client.HTTPConnection.connect` and
  `.request`, `ftplib.FTP.connect` and `ftplib.FTP(host)`,
  `smtplib.SMTP(host)`, `ssl.get_server_certificate`, `webbrowser.open`. The
  capability is in the call that connects: constructing a client object
  without a host (`socket.socket()`, `http.client.HTTPConnection(host, port)`,
  `ftplib.FTP()`, `xmlrpc.client.ServerProxy(uri)`) opens nothing.

## 4. TensorFlow op table (closed)

For TensorFlow graphs the table is the policy: an op listed here has the
capability listed, and an op not listed has none, wherever it appears.

| Capability | Ops |
|---|---|
| (A) code execution | `PyFunc` `PyFuncStateless` `EagerPyFunc` |
| (C) file access | `ReadFile` `WriteFile` `MatchingFiles` `MatchingFilesDataset` `ImmutableConst` `WholeFileReader` `WholeFileReaderV2` `TextLineReader` `TextLineReaderV2` `FixedLengthRecordReader` `FixedLengthRecordReaderV2` `TFRecordReader` `TFRecordReaderV2` `LMDBReader` `LMDBDataset` `ExperimentalLMDBDataset` `TextLineDataset` `TFRecordDataset` `FixedLengthRecordDataset` `DatasetToTFRecord` `ExperimentalDatasetToTFRecord` `SaveDataset` `SaveDatasetV2` `LoadDataset` `SnapshotDataset` `SnapshotDatasetV2` `LoadAndRemapMatrix` `GenerateVocabRemapping` `Save` `SaveSlices` `Restore` `RestoreSlice` |
| (D) network | `DataServiceDataset` `DataServiceDatasetV2` `DataServiceDatasetV3` `DataServiceDatasetV4` `RegisterDataset` `RegisterDatasetV2` `DistributedSave` |
| never a finding | `SaveV2` `RestoreV2` `MergeV2Checkpoints` `ShardedFilename` `ShardedFilespec` (the standard checkpoint saver that every real SavedModel carries), and every op not listed above |

Op names are exact and case-sensitive. Node names, device strings, attribute
values and tensor contents carry no capability.

## 5. Keras

A `Lambda` layer whose `function` carries serialized bytecode is (A):
`"function": {"class_name": "__lambda__", "config": {"code": "<base64>", "defaults": ..., "closure": ...}}`; so is a `Lambda` whose `output_shape` carries bytecode in the same form (Keras
calls it when it rebuilds the model), and so is any layer whose `activation`
is a `__lambda__` dict (in the generic form Keras writes for a lambda,
`{"class_name": "__lambda__", "config": {"value": ["<base64>", defaults, closure]}}`),
since Keras deserializes an activation through the same path. A `Lambda` whose function is a named function reference
(`{"module": "builtins", "class_name": "function", "config": "relu", ...}`),
a named activation, and every other layer type are clean. Layers are found wherever the JSON nests them: `layers`, the `layer` of
a wrapper, and the `config` of a nested model.
