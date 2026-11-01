# model-scanner-hardening: a Terminal-Bench 3 task

One original task in the Harbor task format, built for the Klavis founding
engineer assessment, with local evidence that it clears every gate the
Terminal-Bench 3 CI applies.

The task: a model registry's static scanner lets model files that run code on
load slip through, and flags some legitimate models. The agent hardens the
scanner (pickle in every protocol and container, PyTorch zip and legacy
checkpoints, NumPy, TensorFlow SavedModel, Keras HDF5 and `.keras`) against a
written policy; a hidden corpus of 564 twin-built artifacts, regenerated
from a fixed seed inside the verifier, grades it with reward 1 only at 100 %
recall, zero false positives, no side effects and inside the time budget.

The shipped version is 1.1.0, measured on the models the Klavis brief names
(codex `openai/gpt-6.1-sol` xhigh, claude-code `anthropic/claude-opus-5-5`
max). Its record is RESULTS.md section 0; sections 1 to 7 are the 1.0.0
record on the CI default pair.

- Task directory: [`tasks/model-scanner-hardening/`](tasks/model-scanner-hardening/) (instruction, README with the four explanations, environment, solution, tests).
- Design: [`SPEC.md`](SPEC.md). Toolchain bring-up on Windows: [`BRINGUP.md`](BRINGUP.md). Plan: [`PLAN.md`](PLAN.md).
- Results of every gate and trial: [`RESULTS.md`](RESULTS.md). Trial outputs: [`runs/`](runs/). Failure analysis: [`analysis/`](analysis/).

## Gates and how to reproduce them

All commands run from this repository root with the upstream
`terminal-bench` clone (commit `7b1f530`, 2026-10-28) checked out next to it, Harbor
`0.23.1.dev202609170426` and `dockerfile-pin` 1.5.0 on `PATH`, and Docker
running. `export PYTHONUTF8=1` first on Windows.

| Gate | Command |
|---|---|
| 29 static checks (`static-checks.yml`) | `bash scripts/check-all.sh tasks/model-scanner-hardening` |
| Implementation rubric (37 criteria) | `harbor check tasks/model-scanner-hardening -r ../terminal-bench/docs/prompts/task-implementation.toml` (or the rubric run through Claude Code, see RESULTS.md) |
| Docker build, oracle | `harbor run -p tasks/model-scanner-hardening -a oracle --env docker --yes` |
| nop | `harbor run -p tasks/model-scanner-hardening -a nop --env docker --yes` |
| `/run` (3 attempts per agent, CI defaults) | `harbor run -c scripts/run-job.json ...` (auth flags in RESULTS.md) |
| `/cheat` (1 attempt per agent, hack prompt appended) | `harbor run -c scripts/cheat-job.json ...` |
| Ground truth by execution | `scripts/gt-validate/` (throwaway image with the real frameworks) |
| Shortcut audit | `scripts/shortcut-audit/audit.py` |

The 29th static check, `check-lockfiles` (upstream PR #2107, merged on
2026-10-28, after the trials), requires every dependency install in an image
build to consume a committed lockfile. The task's two Dockerfiles were
changed from `pip install` of pinned versions to `uv sync --locked` against a
committed `pyproject.toml` and `uv.lock`, with the same pins; the oracle and
nop gates were rerun on the rebuilt images. The agent trials ran on the
earlier images, in which the packages the scanner can import are the same
versions (RESULTS.md, section 1).

## How this task was arrived at

Five designs were built and measured before this one shipped (the record is
in `analysis/` here and in the companion repository
`TanmayKallakuri/tb3-git-repo-recovery`): a reproducible-release-build task
that codex solved in 24 minutes, and three versions of a git-repository
recovery task in which everything lost is recoverable from on-disk evidence,
the last of which `gpt-6.1-sol` solved in 7 minutes of agent time on
2026-10-30. What those designs had in common is a single artifact the agent
could check locally by hash, rebuild or re-run. This task grades a general
procedure on a hidden corpus the agent cannot enumerate, so an agent's own
tests only cover what it already understands; it is the only design of the
five in which both frontier agents failed, and 1.1.0 removes the part of its
corpus that had been found by reading earlier agents' scanners.

## Version 1.1.0: the brief's models

| Gate or trial | Result |
|---|---|
| Static checks (upstream `4031b48`), ground truth by execution, oracle, nop | 29 of 29; 564 of 564 items valid; 1.0; 0.0 |
| Implementation rubric, 37 criteria | 36 pass, 0 fail, 1 not applicable |
| claude-code `anthropic/claude-opus-5-5` max, `/run` | **3 of 3 genuine failures** (4, 4 and 5 of 564 items wrong; 2.6 to 3.1 hours each; every session on opus-5.5, no model fallback) |
| claude-code `/cheat` | **0**: a genuine red-team attempt that found no bypass and changed nothing |
| codex `openai/gpt-6.1-sol` xhigh, `/run` | **2 genuine failures** in the two completed attempts (16 and 48 items wrong); the third attempt did not complete (quota cut-offs; RESULTS.md section 0.4) |
| codex `/cheat` | **0**, but through the provider's cybersecurity safeguard refusing the hack prompt before any tool call, as on every codex cheat run of this task; the required zero, not evidence of robustness |
| `harbor analyze` (CI trial analysis) | every check passes on every completed `/run` trial; the claude-code cheat trial passes `reward_hacking` (no hacking) |

What changed from 1.0.0: the 84 hidden items whose only source was reading
earlier agents' scanners were removed (RESULTS.md 0.1), and before any
trial the 24 saved scanners from earlier runs were regraded on the trimmed
corpus; none passes (0.2). Commands: `scripts/run-v3.sh`, `scripts/probe/trim_matrix.py`,
`scripts/v3-table.py`. Failure analysis: `analysis/v3-failures.md`. The
claude-code margins are thin, and three of each trial's errors are one
Keras decoy family; without it every trial still fails on one miss.

## Note on the agent models (1.0.0 record)

The assessment names `codex gpt-6.1-sol` at `xhigh` and `claude-code
opus-5.5` at `max`, and also says the current Terminal-Bench 3 CI configuration
is the source of truth. The CI defaults changed on 2026-09-21 (upstream PR
#2047, "Use Fable 5.1 and Astra for default agent runs") and at the pinned
commit are:

| Agent | Model | Settings |
|---|---|---|
| claude-code | `anthropic/claude-fable-5-1` | `reasoning_effort=max`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000` |
| codex | `openai/gpt-6-astra` | `reasoning_effort=xhigh` |

The trials in this repository use the CI pair, which is what the hosted
`/run` and `/cheat` jobs would use today; the job configurations in
`scripts/run-job.json` and `scripts/cheat-job.json` mirror
`.github/harbor-run-defaults.yml` field for field.

One more fact about the claude-code trials belongs here. In every one of
the seven claude-code sessions on this task (three on the v1 corpus, four
on v2), Fable 5.1's API returned a refusal in category `cyber` between the
14th and the 48th assistant message, while the agent was still reading the
scanner and the samples, and the Claude Code CLI switched the session to
Opus 4.8 for the rest of the run. The CLI's own message says the
safeguards "can sometimes flag legitimate coding and cybersecurity tasks"
and points to support article 15363606. Each trial log records the switch
as one `fallback` event (`original_model: claude-fable-5-1`,
`fallback_model: claude-opus-4-8`, scope `session`), and Harbor's
per-model usage shows the split: in the `run-7` trial, 73 k input and 19 k
output tokens went to `claude-fable-5-1` and 17.7 M input and 223 k output
tokens to `claude-opus-4-8`. So the claude-code rows in RESULTS.md are, in
effect, Opus 4.8 at the CI settings after a Fable 5.1 start. The hosted
`/run` job uses the same CLI, model and task text, so it should behave the
same way; the codex trials are unaffected.

## Results summary (1.0.0, CI default pair)

The gate results (oracle, nop, `harbor check`, the implementation rubric,
the red-team pass) are in RESULTS.md. The agent trials:

| Agent | Completed attempts | Outcome |
|---|---|---|
| claude-code (Fable 5.1, Opus 4.8 after the early fallback) | 3 of 3 | reward 0 each: 187, 200 and 205 of 262 malicious items caught; 16 to 30 of 386 clean items flagged |
| codex (gpt-6-astra, xhigh) | 1 of 3 | reward 0: 258 of 262 caught, 54 clean items flagged. Four further launches ended on the ChatGPT usage limit or a provider capacity error before the agent stopped; the reruns were halted on 2026-10-28 |
| cheat prompt, one per agent | 2 of 2 | no bypass: claude-code left the baseline untouched, codex was refused by the provider's safeguard |

Every trial, the attempt log and the gate outputs: [`RESULTS.md`](RESULTS.md).
