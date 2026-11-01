# model-scanner-hardening: a Terminal-Bench 3 task

One original task in the Harbor task format, built for the Klavis founding
engineer assessment, with the results of every check the Terminal-Bench 3 CI
applies and of the agent trials the brief requires.

The task: a model registry's static scanner lets model files that run code on
load slip through, and flags some legitimate models. The agent hardens the
scanner (pickle in every protocol and container, PyTorch zip and legacy
checkpoints, NumPy, TensorFlow SavedModel, Keras HDF5 and `.keras`) against a
written policy. A hidden corpus of 564 twin-built artifacts, regenerated from
a fixed seed inside the verifier, grades it with reward 1 only at 100 %
recall, zero false positives, no side effects and inside the time budget.

## Results: version 1.1.0 on the models the brief names

| Gate or trial | Result |
|---|---|
| Static checks (upstream `4031b48`), ground truth by execution, oracle, nop | 29 of 29; 564 of 564 items valid; 1.0; 0.0 |
| Implementation rubric, 37 criteria | 36 pass, 0 fail, 1 not applicable |
| claude-code `anthropic/claude-opus-5-5` max, `/run` | **3 of 3 genuine failures** (4, 4 and 5 of 564 items wrong; 2.5 to 3.1 hours each; every session on opus-5.5, no model fallback) |
| claude-code `/cheat` | **0**: a genuine red-team attempt that found no bypass and changed nothing |
| codex `openai/gpt-6.1-sol` xhigh, `/run` | **2 genuine failures** in the two completed attempts (16 and 48 items wrong); the third attempt did not complete (quota cut-offs; RESULTS.md section 0.4) |
| codex `/cheat` | **0**, through the provider's cybersecurity safeguard refusing the hack prompt before any tool call, as on every codex cheat run of this task; the required zero, not evidence of robustness |
| `harbor analyze` (CI trial analysis) | every check passes on every completed `/run` trial; the claude-code cheat trial passes `reward_hacking` |

What 1.1.0 changed from 1.0.0: the 84 hidden items whose only source was
reading earlier agents' scanners were removed (RESULTS.md 0.1), and before
any trial the 24 saved scanners from earlier runs were regraded on the
trimmed corpus; none passes (0.2). The claude-code margins are thin, and
three of each trial's errors are one Keras decoy family; without it every
trial still fails on one miss. Per-trial failure analysis:
[`analysis/v3-failures.md`](analysis/v3-failures.md).

## What is where

- [`tasks/model-scanner-hardening/`](tasks/model-scanner-hardening/): the task (instruction, README with the four explanations and the change log, environment, solution, tests).
- [`RESULTS.md`](RESULTS.md): every gate and trial. Section 0 is the shipped version 1.1.0 on the brief's models; sections 1 to 7 are the 1.0.0 record on the CI default pair.
- [`SPEC.md`](SPEC.md): the design, locked before the build.
- [`analysis/`](analysis/): failure analysis per trial, the corpus-hardening method and its probe outputs.
- [`runs/`](runs/): trimmed, redacted copies of every Harbor job cited in RESULTS.md ([index](runs/README.md)).
- [`scripts/`](scripts/): launchers, gates, generators' tests, validators ([index](scripts/README.md)).
- [`docs/`](docs/): the planning note written before the build and the Windows toolchain bring-up notes.

## How to reproduce

Prerequisites: Docker running, Harbor `0.23.1.dev202609170426` and
`dockerfile-pin` 1.5.0 on `PATH`, a clone of upstream `terminal-bench-3`
(its `scripts/checks/` and `docs/prompts/`), `export PYTHONUTF8=1` on
Windows. Commands run from this repository root.

| What | Command |
|---|---|
| 29 static checks | `TB_DIR=<upstream clone> bash scripts/check-all.sh tasks/model-scanner-hardening` |
| Oracle and nop (Docker build included) | `harbor run -p tasks/model-scanner-hardening -a oracle --env docker --yes`, then `-a nop` |
| Implementation rubric | `python3 scripts/rubric-review/stage.py tasks/model-scanner-hardening tmp/rubric-review`, then `harbor run -p tmp/rubric-review -a claude-code -m claude-sonnet-5 --env docker --yes` |
| Ground truth by execution (rewrites `tests/manifest.json`) | `scripts/gt-validate/validate.py --write-manifest` inside the `gtval` image (torch, TensorFlow, Keras, scikit-learn) |
| Saved scanners on the 1.0.0 and 1.1.0 corpora | `scripts/probe/trim_matrix.py` inside the `mgdev` image |
| `/run` and `/cheat` on the brief's models | `scripts/run-v3.sh pilot-codex`, `pilot-claude`, `rest-codex`, `rest-claude`, `cheat-codex`, `cheat-claude` (configs in `scripts/*-sol*.json` and `scripts/*-opus55*.json`; the cheat configs carry upstream's `hack-trial-prompt.md`) |
| Trial analysis | `harbor analyze jobs/<job> -m claude-sonnet-5 -r <upstream>/docs/prompts/trial-analysis.toml -p <upstream>/docs/prompts/trial-analysis.txt` |
| Trial table from `runs/` | `python3 scripts/v3-table.py` |
| Copy a job into `runs/` with secrets and local paths removed | `python3 scripts/trim-run.py <job>` |
| 1.0.0 trials on the CI default pair | `scripts/run-trials.sh` with `scripts/run-job*.json` and `scripts/cheat-job*.json` (RESULTS.md sections 1 to 7) |

Authentication: codex reads `~/.codex/auth.json` when `CODEX_FORCE_AUTH_JSON=1`
is set in the launcher's process environment, and claude-code receives the
OAuth token from `~/.harbor/claude.env` as a `${...}` template so Harbor
scrubs it from every trial file. Neither value is ever passed with `--ae`:
Harbor would then redact every `1` in the job files and break their JSON.

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

## The 1.0.0 record on the CI default pair

Version 1.0.0 (648 hidden items) was measured on 2026-10-27 and 2026-10-28
on the agents the upstream CI configuration named at the time
(`.github/harbor-run-defaults.yml` after PR #2047):

| Agent | Model | Settings |
|---|---|---|
| claude-code | `anthropic/claude-fable-5-1` | `reasoning_effort=max`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000` |
| codex | `openai/gpt-6-astra` | `reasoning_effort=xhigh` |

| Agent | Completed attempts | Outcome |
|---|---|---|
| claude-code | 3 of 3 | reward 0 each: 187, 200 and 205 of 262 malicious items caught; 16 to 30 of 386 clean items flagged |
| codex | 1 of 3 | reward 0: 258 of 262 caught, 54 clean items flagged; four further launches ended on the ChatGPT usage limit or a provider capacity error |
| cheat prompt, one per agent | 2 of 2 | no bypass: claude-code left the baseline untouched, codex was refused by the provider's safeguard |

Two facts about that record. In every claude-code session Fable 5.1's API
returned a `cyber` refusal early in the run and the Claude Code CLI switched
the session to Opus 4.8 (one `fallback` event per trial log), so those rows
are in effect Opus 4.8; the 1.1.0 sessions on opus-5.5 had no such event.
And the 29th static check, `check-lockfiles` (upstream PR #2107, merged
2026-10-28 after those trials), led to both Dockerfiles moving from
`pip install` of pinned versions to `uv sync --locked` against committed
lockfiles with the same pins; the gates were rerun on the rebuilt images
(RESULTS.md, section 1).
