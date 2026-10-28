# v2 hardening: how the corpus was rebuilt after codex passed v1

## Why

On the v1 corpus (301 items) codex `gpt-6-astra` at `reasoning_effort=xhigh`
reached 108/108 recall with 0 false positives twice (`runs/run-3`). The brief
treats a passing trial as a defect of the task, not a success of the model,
and sends the task back to hardening. The question was what the two
passing scanners had and had not implemented, and whether the policy
document, read as written, already fixed the label of constructs they got
wrong.

## Method

1. **Read the two passing scanners** (`jobs/run-3/.../LMR5moN` and
   `.../Let4Q95`, both kept in `runs/run-3/`). Two independent deep reads,
   each tracing the code path that produces a verdict, each asked only for
   constructs whose label the policy text already fixes. Both scanners
   turned out to be faithful implementations of the document's lists: the
   seed list of callables, the `_compat_pickle` aliases, the enumerated
   reach paths (memo, `getattr`, `attrgetter`, `methodcaller`, `partial`,
   `__import__`, `import_module`), instance methods included. Everything
   outside those lists fell through to "inert object" (scanner A, `_call`)
   or to an unmodelled call (scanner B, `symbolic.py`).
2. **Probe every candidate against every scanner.** `scripts/probe/probe.py`
   builds about 290 candidate artifacts with the corpus generator's own
   writers, each with the label the policy fixes, and runs the oracle, the
   baseline and all five agent scanners from the v1 trials on them. Three
   rounds (`analysis/probe/probe-round{1,2,3}.txt`): the first from the
   policy's own principles (capability breadth, consumption, wrappers,
   resolvers, encoded embedded data, structural variants, false-positive
   traps), the second widening each family, the third from the two code
   reads.
3. **Keep what the policy fixes and both scanners miss; drop what the
   policy does not fix.** Dropped on purpose: polyglots the loader table
   leaves ambiguous (a torch zip with an extra member, an `.npz` with a
   non-`.npy` member, a pickle with a trailing empty zip record), oversized
   `FRAME` desynchronisation (an implementation detail of the C
   unpickler's buffering), classes created with `type()` whose special
   methods fire on collection, `__del__` and `__hash__` hooks, special
   methods stored in an instance `__dict__`, Windows-only or Python-2-only
   names (`os.startfile`, `__builtin__.apply`, `execfile`), `logging.
   FileHandler(delay=True)`, `dictConfig` without a version, short-circuited
   consumers, loaders handed a path they cannot open, `pdb.run` (which
   needs a terminal to execute anything). Also dropped: compile-only
   constructs as clean items, because the seed list names `builtins.compile`
   under (A) and the corpus must not contradict it.
4. **Say in the policy what the corpus relies on.** Every construct kept
   is either a stdlib callable with the same capability as a seed-list entry
   (the document already says "anything with the same capability counts") or
   a route the document now names explicitly as one of its "examples, not a
   list". The semantics the traps rely on were added to section 3: nested
   loads read one stream; BUILD writes `__dict__` when there is no
   `__setstate__` (a module's attributes included); dotted names resolve
   only at protocol 4+; REDUCE's arguments must be a tuple for the C
   unpickler; `pickle.load` stops at a persistent id; argument-less
   `INST`/`OBJ` call a non-class callable; a lazy iterator never consumed
   executes nothing. Two errors in the v1 text were fixed: `platform.popen`
   does not exist in Python 3 and left the seed list; "an `Activation`
   layer is clean" became "with a named activation"; and `Lambda.
   output_shape` bytecode was added to section 5.
5. **Regenerate, validate by execution, re-pin, and run the old scanners
   on the new corpus** (`scripts/probe/corpus_matrix.py`,
   `analysis/probe/corpus-matrix-v2.txt`).

## What the probes found

Round 1 (213 items): both codex scanners wrong on 17, among them
`codecs.open` and `nntplib.NNTP` (breadth), `deque(map(...))` and
`bytes(map(...))` (consumers they did not model), `functools.lru_cache` and
`functools.cache` wrappers, `copy.copy` of a function, `pkgutil.resolve_name`,
`vars()` and `object.__getattribute__`, `ThreadPoolExecutor.submit`, and the
clean twin of a `sys.modules` lookup (both flag any use of `sys.modules`).
Round 2 (77 more): `threading.Timer`, `Thread.run`, `executor.map`,
`Counter(map)`, `dict.fromkeys(map)`, `functools.singledispatch` and
`functools.wraps`, `smtplib.LMTP`, `build_opener().open`; scanner B also
flags the clean twins of several consumers. Round 3 (88, from the code
reads): 27 constructs that all five agent scanners get wrong, including a
module reached through another module's attribute (`shutil.os`), dotted
globals at protocol 4+, `copyreg._reconstructor` with a real base class,
`unittest.mock.Mock` with a side effect, `weakref.finalize` on a temporary,
`code.InteractiveConsole`, `logging.handlers.SysLogHandler`,
`webbrowser.GenericBrowser.open`, `doctest.run_docstring_examples`,
`cProfile.Profile.runcall`, `shelve.Shelf` item access, a module rebound by BUILD,
NEWOBJ-built `methodcaller`/`attrgetter`, `types.FunctionType` reached
through `type()`, `pydoc.locate`, `len.__self__`, `importlib.reload`,
`ssl.create_connection`, `tarfile.bltn_open`, `_thread.start_new_thread`;
and, on the clean side, a persistent id before a payload, a dotted global
at protocol 2, `ftplib.FTP().login()`.

The oracle was wrong on about 40 of the probe items before it was fixed
(`sorted`/`max` with a `key` passed through a partial's state, `copy.copy`,
`vars()`, dotted names, single-stream nested loads, unconsumed iterators,
module rebinding, `type()` of a function, and the construct-but-never-touch
traps among them); the fixes and their unit tests are in
`solution/modelgate/` and `scripts/oracle-tests/`.

## The v2 corpus

648 items: 262 malicious and 386 clean; 197 pickle gadget pairs (the 42
v1 gadgets plus 136 new ones) across the same 15 placements, 68 TensorFlow
and 61 Keras items, 40 structural `X-*` items and 85 benign-only `B-*`
items; family list in SPEC.md section 5.2 and 5.3. Scan time with the
oracle is about 35 s in the verifier (0.15 s per artifact, slowest 0.8 s) of
the 900 s budget. Four draft items were dropped after validation by
execution: a nested `torch.serialization.load` (torch 2.8 defaults to
`weights_only=True`, so it executes nothing), two numpy re-export names that
do not exist in numpy 2.2, and a raw `_thread` whose touch raced the
interpreter's exit; three others were corrected (the `commands` alias only
exists below protocol 3, `functools.wraps` needs a Python-level wrapper, an
unbound `Bdb.run` needs its instance).

The five v1 agent scanners, unchanged, on the 650-item draft of v2:

| Scanner | Wrong | Misses | False positives |
|---|---|---|---|
| codex A (`LMR5moN`, passed v1) | 74 | 57 | 17 |
| codex B (`Let4Q95`, passed v1) | 86 | 57 | 29 |
| claude-code 1 (`VgkkU5J`) | 136 | 96 | 40 |
| claude-code 2 (`RmD6znL`) | 148 | 115 | 33 |
| claude-code 3 (`hthYiEA`) | 122 | 106 | 16 |

The two codex scanners fail across every new family (P12 to P18) and on
the false-positive traps (`B-reduce-list-args`, `B-persid-then-payload`,
`B-dotted-global-proto2`, `B-ftp-login-nohost`, `B-sys-modules-fspath`,
`KERAS-benign-decoy-key`). That is the measurement the trials on v2 start
from; the trials themselves (Part 2 of `analysis/run-trials.md`) are the
test.
