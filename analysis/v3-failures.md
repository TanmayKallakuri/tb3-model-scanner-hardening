# Failure analysis, version 1.1.0 (gpt-6.1-sol xhigh and opus-5.5 max)

Five genuine failures and one genuine cheat trial; four more launches ended
on quotas (three on the ChatGPT usage limit, one on API credit) and are not
counted, and the codex cheat trial was a provider refusal (RESULTS.md,
section 0.4).
Item-level verdicts are in each trial's `verifier/reward-details.json` under
`runs/v3-*`.

## What the two agents did

Both read the policy, the samples and the baseline, rewrote the scanner as a
static interpreter over the pickle opcodes with container recursion and
TensorFlow and Keras walkers, and built their own regression sets from the
policy text. Both stopped believing they were done. Codex's final messages
report "Validation passed" on its own tests (27 and 43 regression tests, a
generated set of about 560 artifacts). The claude-code attempts ran for 2.5
to 3 hours each and measured themselves on their own corpora (for example
274 of 274 malicious caught). Neither had a way to see what its own model of
the policy left out, so their errors line up with their blind spots, not
with randomness.

## Where they failed

| Trial | Misses | False positives | Shape of the failure |
|---|---|---|---|
| codex `pZmdSZW` | 9 | 7 | Breadth: network and file capability through stdlib modules not on the seed list (`poplib`, `POP3_SSL`, `imaplib`, `IMAP4_SSL`, `nntplib`, `dbm.open`, `codecs.open`) and two consumers of a lazy `map` (`dict.fromkeys`, `Counter`). Precision: `importlib.reload` of a harmless module, a `singledispatch` and a `memoryview` twin, a Python-2 style numeric NumPy header, the three Keras decoy keys. |
| codex `ZcCeMyV` | 2 | 46 | Failed closed. It flagged the clean twin of almost every stdlib callable it did not model (sockets, `ftplib`, `smtplib`, `tempfile`, `shutil`, `pathlib`, `zipfile` and `tarfile` writes, `os.popen`, `os.spawnvp`), so recall nearly held while precision collapsed. Misses: `dict.fromkeys(map(...))` and `object.__getattribute__` reaching `os`. |
| claude-code `PiWwP3W` | 1 | 3 | `urllib.request.build_opener()` followed by `.open(url)` inside an npz: a two-step network chain. The three Keras decoy keys. |
| claude-code `XoNXEvQ` | 1 | 3 | `object.__getattribute__(module, "system")` resolved at run time, a reach path the agent named as a known gap in its own notes. The three Keras decoy keys. |
| claude-code `qAQpeTC` | 1 | 4 | `tempfile.TemporaryDirectory` inside a polyglot zip. A `memoryview` twin. The three Keras decoy keys. |

The difficulty the task intends shows in both directions. A scanner that
models a closed list of callables misses the long tail of stdlib modules
with the same capability. A scanner that fails closed on anything unmodelled
flags the clean twins, which differ from the malicious items only in the one
callable. Codex took each route once. Claude-code got closest, at 4 or 5
items of 564, and always on reach paths that go one step past the
policy's named examples.

## The family to watch

`KERAS-benign-decoy-key` accounts for three of the four or five errors in
every claude-code trial, and appears in two codex trials. Each item is a
Lambda-shaped dict with marshaled bytecode placed under `config.notes`, a key
Keras never deserializes. The policy defines a verdict by what loading
executes, lists the places layers live (`layers`, a wrapper's `layer`, a
nested model's `config`), and says for TensorFlow that names and attribute
values carry no capability. A reviewer could still read the decoy as a trap
rather than a test of the policy. With that family removed, every claude-code
trial still fails on the one miss in the table, so no reward changes. But
the claude-code margins are thin, and a later claude-code model could clear
the remaining item.

## What the analyzer said

`harbor analyze` (CI's trial analysis, `claude-sonnet-5`) passed
`task_specification`, `reward_hacking`, `difficulty_crux` and `refusals` on
every trial. Its `difficulty_crux` notes name the same crux as the task
README: zero misses and zero false positives at once, over twins that differ
only in the executed capability. A `near_miss` fail on `ZcCeMyV` in a first pass was contradicted by its
own explanation and became a pass on the rerun; the verdicts vary between
runs (RESULTS.md, section 0.5).
