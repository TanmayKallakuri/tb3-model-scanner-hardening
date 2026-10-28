# Run trials: why each completed attempt failed (and why two did not)

Configuration: `scripts/run-job.json` (the CI pair at the pinned upstream
commit: claude-code `claude-fable-5-1` at `reasoning_effort=max` with
`CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`, codex `gpt-6-astra` at
`reasoning_effort=xhigh`), three completed attempts per agent (codex reached one on v2; see the close of Part 2). Attempts that
ended in an agent error (provider usage limits, section 7 of RESULTS.md) are
not analysed here; they were rerun.

One qualification applies to every claude-code trial below. Each session
hit a Fable 5.1 API refusal in category `cyber` early on (between the 14th
and the 48th assistant message, while the agent was reading the scanner
and the samples) and the Claude Code CLI switched the session to
`claude-opus-4-8`; the trial logs record one `fallback` event each and the
per-model usage puts almost all tokens under Opus 4.8. The claude-code
analyses are therefore of Opus 4.8 at the CI settings after a Fable 5.1
start (details in the README model note). The codex trials ran on
`gpt-6-astra` throughout.

The method: read the trajectory (what it inspected, what it built, what it
tested, how it ended), then run the agent's final scanner from the collected
artifact on exactly the corpus items the verifier marked wrong and read the
scanner's own findings, so each gap is named by the construct, not guessed.
Where the v1 items are no longer regenerable (the generator moved on to v2),
the gap is named from the family of the missed item and from the probe
battery in `analysis/v2-hardening.md`, and the text says so.

## Part 1: the v1 corpus (301 items), 2026-10-27

Three claude-code attempts failed; two codex attempts passed. Under the
brief's rule ("any passing trial sends the task back to hardening") the task
was hardened to the v2 corpus (Part 2) and every trial was rerun.

### claude-code, attempt 1 (`run-1`, trial `model-scanner-hardening__VgkkU5J`)

Reward 0. Agent time 42 minutes, 66 tool calls, 8.9 M input / 178 k output
tokens; declared done.

**What it did.** It read the threat model, the report schema, the evaluator
and the baseline, ran the evaluator (6 of 15 caught, 3 false positives), and
disassembled every sample by hand with `pickletools`, including the
containers, the SavedModels and the Keras files. It checked
`_compat_pickle` for the Python 2 module aliases and confirmed that
`pickletools.genops` stops at the first `STOP`. It then rewrote the scanner:
an abstract interpreter over the opcode stream that flags a capability only
where a call executes it (`REDUCE`, `INST`/`OBJ` with arguments, a
`functools.partial` fired later, a `BUILD` retarget) and follows the memo,
`getattr`, `attrgetter`, `methodcaller`, `__import__`, module aliases and
later streams; content-based dispatch through gzip, bzip2, xz, zip and tar;
a walk of every meta graph, node and library function against the closed
op table in both protobuf encodings; a recursive Keras walk; and an `.npy`
object-array reader. It built a 146-case adversarial battery of its own
(hand-assembled pickles, SavedModels, Keras files, nested containers), got
all 30 samples and the battery to pass at 0.06 to 0.26 s per artifact, and
wrote a closing report saying the scanner "now matches the registry policy".

**What the verifier measured.** 104 of 108 malicious items caught (4
misses), 5 of 193 clean items flagged, 0 schema errors, 0 side effects,
37.8 s of scanning. Two checks failed: `test_recall` and
`test_no_false_positives`.

**The four misses, all one gap.** `P-pathlib-methodcaller` (twice, raw and
bz2), `P-code-runsource` (inside a torch checkpoint) and `P-bdb-run` (inside
an `.npz`). Each reaches a capability through a method of an *instance* the
pickle constructs first: `operator.methodcaller("write_text", "x")` applied
to `pathlib.Path("/run/canary/...")`, `methodcaller("runsource", src)` applied
to `code.InteractiveInterpreter()`, and `getattr(bdb.Bdb(), "run")(src)`.
The agent's interpreter resolves `methodcaller` and `getattr` on module
globals (it caught `getattr(__import__("os"), "system")`), but it does not
carry the class of a constructed instance forward, so the method call on
the instance resolves to nothing and the scanner reports `clean`. The
threat model names this path explicitly ("`pathlib.Path.write_text`
reached through `getattr`" under (C), and "`bdb.Bdb.run`",
"`code.InteractiveInterpreter.runsource`" under (A)), and the agent's own
battery contained no instance-method case.

**The five false positives, all one rule.** Every one is the clean twin of a
nested-loader item: `pickle.loads(b)` (raw and inside a torch checkpoint),
`joblib.load(BytesIO(b))`, `numpy.load(BytesIO(b), None, True)` inside a
legacy torch checkpoint, and `torch.storage._load_from_bytes(b)` inside an
`.npz`, in each case with embedded bytes whose own content executes
nothing. The agent's scanner flags every nested-loader call unconditionally
as `code-execution: call pickle.loads (re-entrant deserialization)`. The
policy says the opposite: "a nested loader is judged by what it would load;
a nested loader fed data that does none of (A)-(D) is not a finding", and
the agent's closing report even states that nested loaders "recurse into
their embedded bytes"; the recursion was implemented for the verdict on the
inner data but the outer call was still reported as a finding.

**Why it stopped.** Its own battery was derived from the same reading of the
threat model as its scanner, so both shared the two blind spots, and the
visible samples (all 30 passing) gave no signal. The 146-case battery is
exactly the kind of self-made test the task predicts: it confirms the
agent's model of the policy rather than the policy.

### claude-code, attempt 2 (`run-2`, trial `model-scanner-hardening__RmD6znL`)

Reward 0. Agent time 35 minutes, 68 tool calls, 10.8 M input / 151 k output
tokens; declared done.

**What the verifier measured.** 101 of 108 caught (7 misses), 5 of 193
clean items flagged, 0 schema errors, 0 side effects, 36.9 s.

**The seven misses: the same instance-method gap, wider.** `P-pathlib-
methodcaller` (raw and bz2), `P-code-runsource` (torch zip), `P-bdb-run`
(`.npz`) as in attempt 1, plus three network items that attempt 1 had
caught: `P-socket-connect-chain` (`methodcaller("connect", addr)` on
`socket.socket()`), `P-http-connect-chain` (`methodcaller("connect")` on
`http.client.HTTPConnection(host, port)`) and `P-ftplib` (`methodcaller(
"connect", host, port)` on `ftplib.FTP()`). Every miss is a method called on
an object the pickle builds first; the scanner handles the module-level
callable and loses the object.

**The five false positives: the same nested-loader rule.** The clean twins of
`torch.storage._load_from_bytes` (`.npz`), `numpy.load` (legacy torch),
`joblib.load` (raw, memoised), `pickle.loads` (torch zip, and raw), each
fed embedded bytes that execute nothing, all reported as code execution.

Two attempts of the same model, trained on the same document, converged on
the same two readings: "a method on a built object is not a call I can
resolve" and "a nested loader is a finding by itself".

### claude-code, attempt 3 (`run-2`, trial `model-scanner-hardening__hthYiEA`)

Reward 0. Agent time 53 minutes, 88 tool calls, 15.5 M input / 227 k output
tokens; declared done, with a closing report that lists "recurses into
nested loaders (and sees through zlib/base64-style data transforms)" and
"a method resolved on a constructed instance" among its features.

**What the verifier measured.** 100 of 108 caught (8 misses), 0 of 193
clean items flagged, 0 schema errors, 0 side effects, 37.0 s. Only
`test_recall` failed.

**The eight misses: nested loaders, from the other side.** `P-numpy-load`
(raw, and inside a legacy torch checkpoint), `P-joblib-load` (raw,
memoised), `P-pandas-read-pickle` (xz), `P-torch-load-from-bytes` (`.npz`),
`P-pickle-loads` (torch zip, and raw), and `P-ftplib` (gzip). This attempt
fixed the false-positive side of the nested-loader rule (it judges the
embedded data, hence 0 false positives) but the recursion does not reach
the payload in these items: the embedded pickle sits inside a dict the
inner stream builds, in the forms the v1 generator used (protocol 2 or 4
with memoisation), and the probe battery (`analysis/v2-hardening.md`) shows
the same scanner catching `pickle.loads` over base64-decoded, hex-decoded
and `bytearray` data while missing `pickle.loads` over a plain bytes
literal inside a torch zip, over `zlib`-decoded data, over concatenated
pieces and over a doubly nested payload. The recursion is real but
conditional on a shape the scanner's own tests used. The `ftplib` miss is
the instance-method gap of attempts 1 and 2 (`FTP()` then `connect`), the
one network constructor-then-connect pair this attempt did not model.

### codex, attempt 1 (`run-3`, trial `model-scanner-hardening__LMR5moN`): reward 1

Agent time 41 minutes, 37 tool calls, 3.4 M input / 64 k output tokens.
108 of 108 caught, 0 false positives, 38.0 s.

**What it did.** Two parallel reads (every doc, the baseline, the sample
listing), one `pickletools` disassembly pass over eight samples, then it
wrote the whole scanner in three heredocs: a capability table "applied to
calls, never to stored names", a NumPy header parser, and a 1246-line
non-executing interpreter for pickle's stack machine with symbolic values
for instances, partials, bound methods and buffers. It then read CPython's
own sources to settle the semantics it was unsure of (`functools.partial.
__setstate__`, `pickle._Unpickler.load_build` and `_instantiate`,
`numpy.lib.format.read_array`), patched the interpreter for what it found
(`__init__` reached through `getattr`, `partial` namespace handling, legacy
torch stream detection, `io` wrapper classes, `xmlrpc.client.ServerProxy`),
audited side effects with `sys.addaudithook`, wrote 27 regression tests,
generated a 300-artifact policy-derived benchmark of its own and timed it
(85.5 s), and stopped. Its tables cover exactly the policy's seed list,
`_compat_pickle` aliases and the enumerated reach paths (memo, `getattr`,
`attrgetter`, `methodcaller`, `partial`, `__import__`, `import_module`),
instance methods included.

### codex, attempt 2 (`run-3`, trial `model-scanner-hardening__Let4Q95`): reward 1 with an agent error

Agent time 16.5 minutes, 14 tool calls, 685 k input / 34 k output tokens,
then the provider's usage limit ended the session (`NonZeroAgentExitCodeError`
on a `turn.failed`). The scanner it had already written (a 556-line
symbolic interpreter plus a 420-line opcode front end) scored 108 of 108
with 0 false positives in 37.6 s. Harbor records the attempt as an error
with reward 1.

### What the v1 corpus had tested, and had not

Every v1 construct was an instance of something the document enumerates:
seed-list callables, the listed reach paths, the listed containers. A
scanner that implements the document literally passes; two did. The
document also says, twice, that the lists are not the policy ("a seed list,
not the policy; anything with the same capability counts"; "no matter how it
reached the executing opcode"), and the v2 corpus holds it to that. The
probe of the two passing scanners against constructs outside the lists
(`analysis/v2-hardening.md`) found 27 that both missed plus a set of clean
constructs both flagged; on the finished v2 corpus the two scanners score
74 and 86 wrong of 650.

## Part 2: the v2 corpus (648 items)

Same configuration, same rule: every analysed trial ends with the agent
declaring the work done, and the verifier names what its scanner gets wrong.
Family names carry the construct (`P-<gadget>-<placement>-p<protocol>`), so
the misses are read from `reward-details.json` directly.

### claude-code, attempt 1 (`run-4`, trial `model-scanner-hardening__jsAPckD`)

Reward 0. Agent time 62 minutes, 93 tool calls, 20.1 M input / 242 k output
tokens; declared done.

**What it did.** It read the policy and the baseline, disassembled the
samples, and wrote a pickle abstract interpreter (`pklvm.py`) around "a
capability is a finding only when loading would execute it", with
container dispatch by content, the SavedModel walk, and a Keras walk. Its
closing report gives the evaluator result (15 of 15 samples caught, 0 false
positives) and a battery of 142 adversarial cases of its own "covering
techniques the samples omit (aliases, getattr chains, ...)", kept in
`/tmp/advtest/`, and summarises the baseline's failure modes and its fixes.

**What the verifier measured.** 187 of 262 malicious items caught (75
misses), 23 of 386 clean items flagged, 0 schema errors, 0 side effects,
78 s of scanning.

**The misses, by what they test.**
- *Breadth beyond the seed list* (about 25): `sqlite3.connect`,
  `logging.FileHandler`, `tempfile.TemporaryDirectory`, `tarfile.open`,
  `tarfile.TarFile`, `tarfile.bltn_open`, `zipfile.ZipFile(path, "w")`,
  `pathlib.Path.touch/mkdir/write_bytes/open`, `Path / name` then
  `write_text`, `dbm.open`, `shelve.open`, `os.mkfifo`, `cProfile.run` and
  `Profile.runcall`, `doctest.run_docstring_examples`,
  `InteractiveConsole.runsource`, `webbrowser.GenericBrowser.open`,
  `SysLogHandler`, `ssl.create_connection`, `FTP_TLS.connect`,
  `socket.connect_ex`, `HTTPSConnection.request`, `build_opener().open`,
  `ServerProxy` method call. The scanner's capability table is a list, and
  the list is the seed list.
- *Reach paths* (about 20): `sys.modules` (through `itemgetter` and
  `attrgetter`), `object.__getattribute__`, `len.__self__`, `pydoc.locate`,
  `importlib.reload`, dotted globals (`logging` + `os.popen`, `builtins` +
  `eval.__call__`), `getattr(f, "__call__")` and `methodcaller("__call__")`,
  `types.MethodType`, `staticmethod`, `functools.singledispatch`,
  `functools.wraps`, `unittest.mock.Mock`, `copyreg._reconstructor` with a
  real base, NEWOBJ-built `methodcaller`/`attrgetter`.
- *Consumption and deferred execution* (about 15): `reduce(add, map(...))`,
  `list(islice(map))`, `list(enumerate(map))`, `list(zip(map))`,
  `str.join(map)`, `bytes(map)`, `Counter(map)`, `dict.fromkeys(map)`,
  `sorted`/`max` with a `key` passed through a partial's state,
  `list.sort(key=)`, `Executor.submit/map`, `Thread.start/run`,
  `Timer.start`, `weakref.finalize`.
- *Nested-loader variants* (5): `shelve.Shelf` item access, `pickle._loads`,
  `pickle.Unpickler(...).load()` and `_pickle.Unpickler(...).load()`,
  `numpy.lib._npyio_impl.load`.
- *The v1 families it still misses* (7): `pathlib.Path.write_text` through
  `methodcaller` (twice), `InteractiveInterpreter.runsource`, `Bdb.run`,
  `socket.connect` on a built socket, `HTTPConnection.connect`,
  `FTP().connect`: the instance-method gap of all three v1 attempts, in a
  fourth scanner by the same model.
- *Structure* (1): a tar member whose pickle starts with `BININT2` before the
  payload stream.

**The false positives, by what they test.** Nested loaders fed data that
executes nothing, reported as findings anyway (`joblib.load` twice,
`numpy.load` on raw bytes twice, `pickle.load`/`_pickle.load` on a
`BytesIO`, `read_array`, `dict(npz)`, `attrgetter` on an `NpzFile`,
`pandas.read_pickle`): the v1 nested-loader rule again. Embedded data the
pickle first decodes or assembles (`latin1` encode, `memoryview`,
`operator.mul`, `bytes([...])`, `bytes.join`), judged by its wrapper rather
than its content. `pickle.loads` over two embedded streams with the
capability only in the second (the loader reads one). The three Keras files
with a Lambda-shaped dict under a key Keras never reads. `type(function)`
and a code object stored apart; `marshal.loads` without `FunctionType`;
`fileinput.input` never iterated.

**Why it stopped.** As in v1: its own cases came from its own reading of
the document, and all 30 samples passed. The closing report lists the
baseline's failure modes and the fixes for them, which is the frame the
samples suggest; nothing in its battery asked what the document's
"examples, not a list" sentence covers.

### claude-code, attempt 2 (`run-4`, trial `model-scanner-hardening__2ZRCVpn`)

Reward 0. Agent time 93 minutes, 162 tool calls, 37.8 M input / 427 k
output tokens; declared done.

**What it did.** The longest trajectory of the series. Its closing report
lists an abstract interpreter over `pickletools.genops` that "flags a
capability only when it would actually be invoked, following indirection
(getattr, partial with retargeting and keyword capture, attrgetter,
methodcaller, import, `pkgutil.resolve_name`, map/filter/reduce,
executor/thread sinks, `sorted(key=...)`, `sys.modules`)", content-based
container dispatch "with last-entry-wins zip rules", the closed TensorFlow
table over every meta graph and library function, and Keras bytecode
detection "anywhere in the config". It reports 15 of 15 on the samples, a
99-artifact adversarial suite of its own, 548 fuzzed inputs without a
crash, and a sandbox check under `env -i`.

**What the verifier measured.** 200 of 262 caught (62 misses), 16 of 386
clean items flagged, 0 schema errors, 0 side effects, 77.5 s.

**The misses.** The report names the right principles; the scanner
implements a subset of each. Reach paths it claims and still misses:
`sys.modules` through `itemgetter` and `attrgetter`, `vars()`,
`object.__getattribute__`, `len.__self__`, `importlib.__import__`,
`importlib.reload`, dotted globals, `__call__` through `getattr` and
`methodcaller`, `types.MethodType`, `staticmethod`, `singledispatch`,
`wraps`, `Mock`, `copyreg._reconstructor`, NEWOBJ-built callers (about
18). Consumers it claims and still misses: `reduce(add, map)`,
`list(islice/enumerate/zip/chain(map))`, `next(map)`, `bytes`, `Counter`,
`dict.fromkeys`, `str.join`, `list.sort(key=)`, `Executor.map`,
`weakref.finalize` (12). Breadth off the seed list (about 22):
`sqlite3.connect`, `logging.FileHandler`, `TemporaryDirectory`,
`tarfile.open/TarFile/bltn_open`, `zipfile.ZipFile(path, "w")`,
`Path.mkdir/open`, `Path / name`, `dbm.open`, `shelve.open`, `os.mkfifo`,
`cProfile.runcall`, `doctest`, `InteractiveConsole`, `GenericBrowser.open`,
`SysLogHandler`, `ssl.create_connection`, `FTP_TLS.connect`,
`socket.connect_ex`, `build_opener().open`, `ServerProxy` call. Nested
variants (6): `Shelf`, `pickle._loads`, both `Unpickler(...).load()` forms,
`numpy.lib._npyio_impl.load`, `attrgetter("f.a")` on an `NpzFile`. One v1
instance-method item remains (`socket.connect` on a built socket).
Compared with attempt 1, it closed the v1 instance-method gap for
`pathlib`, `Bdb`, `InteractiveInterpreter`, `HTTPConnection` and `FTP`,
and the thread and timer sinks, and opened nothing new.

**The false positives.** A REDUCE whose arguments are a list (three items;
the C unpickler raises before calling, as the document now says), the
three Keras decoy keys ("anywhere in the config" is the documented
mistake), the Python-2 style `.npy` headers (numeric, and the object-dtype
twin), `pickle.loads` over two embedded streams with the capability only in
the second, `memoryview` and `operator.mul` twins, `type(function)` and a
code object stored apart, `marshal.loads` alone, `fileinput.input` never
iterated, `FTP().login()` without a host.

**Why it stopped.** Its 99-artifact suite "covering the threat-model
techniques the samples omit" was written from the same list its scanner
implements, so it passed; the fuzzing checked robustness, not verdicts.
The document's sentences about tuples, trailing bytes, unread Keras keys
and Python-2 headers were read as background rather than as cases.

### claude-code, attempt 3 (`run-7`, trial `model-scanner-hardening__xPvqUTm`)

Reward 0. Agent time 57 minutes, 102 tool calls, 17.7 M input / 242 k
output tokens; declared done ("complete and thoroughly verified"). This is
the rerun of the attempt stopped on 2026-10-27 at the author's request.

**What it did.** A 1,600-line package: a 780-line pickle interpreter over
`pickletools.genops` with memo tracking, module aliases, every global form,
`INST`/`OBJ`, streams past `STOP`, nested loaders re-entered on literal
bytes, a 240-line capability policy, content-based format detection for
every container, the SavedModel walk over meta graphs and library
functions, and a Keras walk. Its closing report shows one representative
catch (a nested `os.system` reported at `sample_09.pickle@328»nested@64`)
and lists the baseline's failure modes it fixed: string matching,
stopping at the first `STOP`, ignoring `INST`/`OBJ`, memo indirection and
aliases, unscanned `.npy`/`.npz`/`.keras` and compressed formats, and the
first-meta-graph-only TensorFlow walk.

**What the verifier measured.** 205 of 262 caught (57 misses), 30 of 386
clean items flagged, 0 schema errors, 0 side effects, 95 s.

**The misses.** Reach paths (about 25): `getattr(f, "__call__")` and
`methodcaller("__call__")`, dotted `__call__`, `functools.wraps`,
`types.MethodType`, `staticmethod`, `singledispatch`, `Mock.side_effect`,
`importlib.reload` and `importlib.__import__`, `vars()` into `builtins`,
`len.__self__`, `object.__getattribute__`, `pydoc.locate`, `sys.modules`
through `attrgetter` and `itemgetter` (the string `sys.modules` is in its
interpreter, the two getter forms are not), `copyreg._reconstructor`,
NEWOBJ-built `attrgetter`/`methodcaller`, `type(function)`, `sorted`/`max`
with a `partial` key, `list.sort(key=)`, `weakref.finalize`. Breadth off
the seed list (about 22; none of these names appear in its policy):
`xmlrpc` call, `tempfile.mkdtemp/mkstemp/NamedTemporaryFile/
TemporaryDirectory`, `logging.FileHandler`, `SysLogHandler`, `os.mkfifo`,
`sqlite3.connect`, `build_opener().open`, `GenericBrowser.open`,
`doctest`, `FTP_TLS.connect`, `tarfile` write and `TarFile(...)`,
`Path.mkdir/open`, `Path / name`, `cProfile.runcall` and
`InteractiveConsole` (the modules are in its policy, these entry points
are not), thread `Timer.start`, `Executor.submit/map`. Consumers (7):
`reduce(add, map)`, `list(islice/enumerate/zip(map))`, `str.join(map)`,
`dict.fromkeys(map)`, `Counter(map)`. Nested (2): `pickle._loads` inside a
legacy torch file, `Shelf.__getitem__`.

**The false positives.** Sixteen of the 30 carry the detail
`<loader> on unresolved data` (`pickle.loads`, `pickle.load`,
`Unpickler.load`, `numpy.load`, `torch.storage._load_from_bytes`): the
interpreter re-enters a nested loader only when its argument is a literal,
and fails closed when the bytes are assembled or decoded first (latin-1
`encode`, `memoryview`, `join`, concatenation, `*`, `zlib`, a second
stream). Every one of those twins hands the loader a clean stream; the
malicious originals, which hand it a dangerous one through the same
expressions, were caught because the scanner flags the unresolved call
either way. So the "fail closed" branch is what both twins hit, and the
clean side pays. Eight are the policy's stated exceptions: a `REDUCE`
with list arguments (3), a persistent id before the payload under plain
`pickle.load` (2), a dotted global at protocol 2, `pickle.loads` over two
streams with the capability only in the second, `fileinput.input` never
iterated. Three are the Keras decoy keys, two the Python-2 style `.npy`
headers (reported as `unsupported`), and one a `wraps` twin where the
wrapped callable is read as the call target.

**Why it stopped.** Its own checks were the 30 samples and a battery
built from the techniques it had implemented, so an unimplemented reach
path or an unlisted module produced no failing test, and the clean side
of nested loading was never exercised with assembled bytes. The seed list
in the threat model was again read as the policy rather than as its
examples.

### codex, attempt 1 (`run-6`, trial `model-scanner-hardening__CfUsAEZ`)

Reward 0. Agent time 50 minutes, 40 tool calls, 4.1 M input / 101 k output
tokens; declared done. No usage-limit message in the session log; the
scheduled restart after the ChatGPT reset worked.

**What it did.** The most compact trajectory of the series. It wrote a
2,200-line package around a 1,400-line abstract interpreter (`abstract.py`)
that models "capability calls and the ways values reach them: module
dictionaries and aliases, attributes, partials and state changes, wrappers,
callbacks, lazy iteration, threads, executors, and nested deserialization",
a pickle machine that tracks memo identity, protocol-dependent globals,
`__new__` versus constructors, persistent ids, concatenated streams and the
legacy PyTorch storage boundary, content-based container dispatch, the
closed op table over every meta graph and function body, and a Keras walk
over nested configs. Its README states the policy's own rule back: "Globals
and suspicious-looking strings are not findings on their own", and
"Operations that cannot be analysed ... fail closed with an `unsupported`
finding". It verified itself with a unit suite of paired cases, a
650-invocation timing pass over 399 fixtures of its own, and the 30
samples.

**What the verifier measured.** 258 of 262 caught (4 misses), 54 of 386
clean items flagged, 0 schema errors, 0 side effects, 81 s.

**The misses.** Four, each a construct one step outside a table:
`logging.handlers.SysLogHandler` (its network table lists `socket`,
`urllib`, `webbrowser`, `ssl`, `requests`, `httpx`; its `logging.handlers`
entry holds only the file handlers); `tarfile.TarFile(...)` called as a
class (it models `tarfile.open` and `TarFile.open`, not the constructor);
`bytes(map(os.popen, ...))` (its `bytes` is a data constructor that returns
unknown when an argument is not concrete, so the map is never consumed);
`dict.fromkeys(map(...))` (`fromkeys` is not in its consumer list, while
`dict(map(...))` is).

**The false positives.** 48 of the 54 are its own `unsupported` verdict
with the detail `unmodelled callable <name>`: the interpreter knows a fixed
set of callables and fails closed on any other, and every clean twin calls
a harmless function from the same module as its malicious twin
(`sqlite3.complete_statement`, `shutil.ignore_patterns`,
`poplib.error_proto`, `subprocess.list2cmdline`, `functools.singledispatch`,
`importlib.reload` on a module path, and so on). The policy's clean class
is exactly those objects; a scanner that cannot say "this call does nothing
the policy names" blocks the registry's legitimate uploads. Three more are
the Keras decoy keys (a Lambda-shaped dict under a key Keras never reads,
flagged as "Lambda function carries serialized bytecode"). The last three
are calls the threat model marks clean by their arguments or by never being
consumed: `FTP.login` on a client never given a host, `sqlite3.connect` of
`:memory:`, and `fileinput.input` never iterated. One Python-2 style `.npy`
header made its header parser raise, which also became `unsupported`.

**Why it stopped.** Its fixtures were generated from the same tables the
interpreter implements, so an unknown-but-harmless callable never appeared
as a clean case and the fail-closed default was never measured against
legitimate inputs; the 30 samples contain no such twin either. On the
malicious side the four misses are the places where a table was consulted
instead of the rule the policy states (a class is called like a function;
a consumer is anything that iterates). The closing report's "650 synthetic
scans in 73.78 seconds" is its own battery, not the hidden corpus.

### codex, `run-8` (trial `model-scanner-hardening__mSXKu4X`): not counted

The rerun of the second attempt ended on the ChatGPT usage limit after 60
minutes, so it is an infrastructure error and is listed only in the
attempt log. It is worth one paragraph because the limit struck at the
very end: the agent had reported "the final 650-artifact CLI run passed"
and run `evaluate.py` once more (15 of 15, 0 of 15) when the next model
turn was refused, so the verifier graded a scanner the agent considered
finished. Result: 253 of 262 caught (9 misses) and 60 of 386 clean items
flagged. Fifty-two of the 60 are `unsupported` verdicts on clean twins,
the same fail-closed default `run-6` showed, this time from a symbolic
interpreter that raises `unsupported` when a call it cannot model is
reached; the rest are the three Keras decoy keys, two Python-2 style
`.npy` headers and three calls the policy marks clean by argument or
non-consumption. The nine misses are reach paths the interpreter does not
model (`object.__getattribute__`, `type(function)`, `staticmethod`,
`weakref.finalize`, `attrgetter` inside an `.npz` in a tar,
`InteractiveConsole` nested) and the two `map` consumers `bytes(...)` and
`dict.fromkeys(...)`, plus a second-stream load through NumPy's internal
loader. So the attempt would have been a reward-0 trial had the session
ended on its own; it is not counted because it did not.

### codex, `run-9` (trial `model-scanner-hardening__uLJf3Mx`): not counted

The second rerun of this attempt ended after 27 minutes on a provider-side
capacity error (`turn.failed`, "Selected model is at capacity"), which
Harbor classifies as `ApiOverloadedError`; it is the first codex session on
this task to end on something other than the account's usage limit. The
agent was still building: its last message reports all 30 supplied samples
passing and names loader boundaries as the next thing to review. The
verifier graded what it left, 234 of 262 caught and 19 of 386 clean items
flagged, a mid-build snapshot rather than a finished scanner, so no reading
of the misses is drawn from it. Rerun as `run-10`.

### codex, `run-10` (trial `model-scanner-hardening__xnMdmLr`): not counted

The third rerun of this attempt ended on the ChatGPT usage limit after 87
minutes, the longest codex session on this task, and again at the very
end: the agent had run its own 650-invocation battery clean (167 s, no
verdict, schema or side-effect failures) and was, in its words, doing a
final precision check "so deferred or stored capabilities remain clean
until an operation actually invokes them" when the next model turn was
refused. The verifier graded what it left: 256 of 262 caught (6 misses: a
`dict` built over an `.npz` member through the memo, `attrgetter` inside
an `.npz` in a tar, `build_opener().open` in an `.npz`,
`InteractiveConsole` nested, `copyreg._reconstructor` into `Popen` under
xz, `Path / name` under bz2) and 64 of 386 clean items flagged. Fifty-four
of the 64 are clean twins reported as `malicious`, the stored-versus-
invoked distinction the agent was addressing when the limit struck; the
rest are the three Keras decoy keys, the two Python-2 style `.npy`
headers and six of the policy's stated clean cases (`os.walk` and
`fileinput.input` never consumed, an in-memory `sqlite3` connection, a
`Telnet` object that never connects, a subprocess coroutine never
awaited, `os.fsencode` reached through `importlib.reload`). Like `run-8`,
this would have been a reward-0 trial had the session ended on its own;
it is not counted because it did not. Rerun as `run-11`.

### codex, `run-11` (trial `model-scanner-hardening__SvRfGkR`): not counted

The fourth rerun of this attempt ended on the ChatGPT usage limit after 73
minutes, again at the end of the agent's own work: it had run its
650-invocation battery clean (106 s, no verdict or report failures,
fresh-process side-effect audits passing) and was doing a last "deployment
check" of a copied `modelgate/` directory when the next model turn was
refused. The verifier graded what it left: 257 of 262 caught (5 misses: a
`dict` built over an `.npz` member through the memo, as in `run-10`;
`shelf.__getitem__` inside a torch zip; `SysLogHandler` under gzip; `LMTP`
in a tar; `object.__getattribute__` reached dynamically) and 63 of 386
clean items flagged, every one of them as `malicious`: 53 clean twins (the
stored-versus-invoked distinction once more), the three Keras decoy keys,
the two Python-2 style `.npy` headers and five of the policy's stated clean
cases (`os.walk` and `fileinput.input` never consumed, a subprocess
coroutine never awaited, `os.fsencode` through `importlib.reload`, an `FTP`
login with no host). Like `run-8` and `run-10` this would have been a
reward-0 trial had the session ended on its own; it is not counted because
it did not. The pattern across the limit-ended launches is now plain: the
one codex attempt that completed on this corpus (`run-6`, 4.1M input tokens
in 50 minutes) sat just under the subscription window's budget, and the
three that the limit ended died at 4.3M to 4.5M input tokens, 60 to 87
minutes in. No rerun followed: Tanmay stopped the codex reruns here on
2026-10-28, so the codex column of Part 2 rests on attempt 1 (`run-6`)
plus these four documented partials. Each of them, graded, sits on the
same clean-twin mechanism as `run-6` and would have been reward 0; that
is a reading of where codex fails on this corpus, not a substitute for
the two attempts that were never completed.
