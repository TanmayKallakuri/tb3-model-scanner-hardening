# Bring-up: Terminal-Bench 3 toolchain on this Windows machine

Done 2026-10-26 (evening, local time; trial timestamps below are UTC 2026-10-27).
Machine: Windows 11 Home 10.0.26300, Docker Desktop 29.8.1 (VM: 16 CPUs, 15.3 GB),
Git Bash (`git version 2.53.0.windows.2`) as the shell for every script here.

Goal of this pass: pinned Harbor on PATH, upstream `terminal-bench` clone, a
one-shot static-check runner, Harbor job configs that mirror the TB3 CI
defaults, and proof that oracle, nop, codex and claude-code trials run on this
box. Nothing here is the task itself.

## 1. Versions installed

| Tool | Version | How installed | Location |
|---|---|---|---|
| harbor (`harbor`, `hb`, `hr`) | `0.23.1.dev202609170426` (matches `.github/harbor-version`) | `uv tool install "harbor==0.23.1.dev202609170426"` | `%USERPROFILE%\.local\bin\harbor` (venv in `%APPDATA%\uv\tools\harbor`) |
| dockerfile-pin | `1.5.0` | `dockerfile-pin_windows_amd64.zip` from the GitHub release, sha256 `6afcddc6...0264827` verified against the release `checksums.txt` (CI pulls the linux tarball, sha `88845bfb...02e95ea6`, same release) | `%USERPROFILE%\.local\bin\dockerfile-pin.exe` |
| terminal-bench clone | commit `6964e92` "Validate verifier reports in hosted task CI (#2316)", `--depth 1`; pulled to `7b1f530` on 2026-10-28 for the new `check-lockfiles` check (section 9) | `git clone --depth 1 https://github.com/harbor-framework/terminal-bench.git` | `%USERPROFILE%\Downloads\terminal-bench` |
| python3 | 3.13.13 (`python3` resolves to the WindowsApps alias, backed by `Python313`) | already present | `python3 -I` imports `tomllib` and `yaml` (PyYAML 6.0.3) |
| uv | 0.11.6 | already present | |
| Docker Compose | v5.5.1 | ships with Docker Desktop | |
| codex-cli | 0.153.3, logged in via ChatGPT (`~/.codex/auth.json` exists) | already present | |
| Claude Code | 2.1.311, logged in via claude.ai | already present | |

## 2. Repo state (`%USERPROFILE%\Downloads\klavis-tb3`)

- `git init` done. **No commits and no git identity set**; Tanmay sets his own.
- `.gitattributes`: `* text=auto eol=lf` plus `*.sh text eol=lf` and a few
  binary patterns. Shell scripts run inside Linux containers; CRLF breaks them.
- `.gitignore`: `jobs/`, `.venv/`, `__pycache__/`, `*.pyc`, `*.env`, `.env*`, `tmp/`.
- `scripts/check-all.sh`: runs all 28 upstream static checks (section 4).
- `scripts/run-job.json`, `scripts/cheat-job.json`: Harbor JobConfigs (section 5).
- `BRINGUP.md`: this file. `PLAN.md`: the orchestrator's plan.
- `tmp/` held the throwaway smoke task (gitignored) and was deleted after the trials.

## 3. Commands run, in order, with outcomes

1. `uv tool install "harbor==0.23.1.dev202609170426"` -> installed 3 executables; `harbor --version` prints exactly `0.23.1.dev202609170426`.
2. `git clone --depth 1 https://github.com/harbor-framework/terminal-bench.git %USERPROFILE%\Downloads\terminal-bench` -> HEAD `6964e92`.
3. Read `.github/workflows/static-checks.yml`: CI downloads `dockerfile-pin` v1.5.0 (linux tarball, sha-checked), installs `PyYAML==6.0.2` into a venv for `python3 -I`, then loops 28 named checks (29 since PR #2107 on 2026-10-28, section 9) with `bash scripts/checks/<check>.sh tasks/<slug>` from the PR root, exporting `TASK_DIRS`, `FIX_DIRS`, `BASE_DIR`, `REPO_URL`, `PR_NUMBER`.
4. Installed `dockerfile-pin` (Windows build, checksum OK) to `~/.local/bin`, which is already on the Git Bash PATH. `dockerfile-pin version` -> `1.5.0`.
5. Found the clone checked out with **CRLF** (global `core.autocrlf=true`, upstream has no `.gitattributes`). Fixed in that clone only:
   `git -C terminal-bench config core.autocrlf false && git rm -r --cached -q . && git reset --hard -q`. All `.sh/.py/.md/.toml/Dockerfile` under `tasks/` are now LF except three files in `hof-topology-interpenetration/solution/` that are CRLF in upstream's own index.
6. `git init` in `klavis-tb3`; wrote `.gitattributes` and `.gitignore`.
7. Wrote `scripts/check-all.sh`; ran it on `terminal-bench/tasks/ctr-optimization` -> **28 passed, 0 warned, 0 failed**, exit 0. (First run failed only `check-task-changelog.sh` because I had marked the unmodified upstream task as "fixed"; the script now derives `FIX_DIRS` the way CI does, see section 4.)
8. `harbor job schema` -> JSON schema saved and read; `harbor agent schema claude-code|codex` confirmed `reasoning_effort` is the kwarg name for both.
9. Wrote `scripts/run-job.json` and `scripts/cheat-job.json`; `harbor run -c <file> --print-config` prints a resolved config for both. The cheat `extra_instructions[0]` is byte-identical to `docs/prompts/hack-trial-prompt.md` (2747 chars, built with `Path.read_text()` exactly like `scripts/ci/hosted_trial_workflow.py`).
10. `harbor tasks init smoke -p tmp/ --org local --author "Smoke Test <smoke@example.invalid>" --description "Smoke test task"` -> scaffolded `tmp/smoke` (the first attempt without `--org` aborted at an interactive prompt; the final check-mark print crashed with `UnicodeEncodeError` but the files were already written). Replaced the vacuous template test with a real check: instruction asks for `/app/hello.txt` containing `hello from harbor\n`; `solve.sh` writes it; `test_outputs.py` asserts the exact content.
11. Smoke trials, all run from `klavis-tb3` with `PYTHONUTF8=1` exported (section 6 table):
    - `harbor run -p tmp/smoke -a oracle --env docker --yes --job-name smoke-oracle -q` -> reward 1.0, 1m02s.
    - `harbor run -p $HOME/Downloads/terminal-bench/tasks/cargo-flight-dispatch -a oracle --env docker --yes --job-name cargo-oracle -q` -> reward 1.0, 1m30s, `verifier_environment_mode = separate` (bonus: proves the `tests/Dockerfile` separate-verifier path builds and runs here).
    - `harbor run -p tmp/smoke -a nop --env docker --yes --job-name smoke-nop -q` -> reward 0.0, 35s.
    - `harbor run -p tmp/smoke -a codex -m openai/gpt-6-astra --ae CODEX_FORCE_AUTH_JSON=1 --ak reasoning_effort=xhigh --env docker --yes --job-name smoke-codex -q` -> reward 1.0, 5m15s, 0 exceptions; the ChatGPT-subscription `auth.json` was accepted and the trajectory records `gpt-6-astra`. The job dir contains no JWT-shaped strings (checked), only a log line naming the auth.json path. **But** the `--ae` value `1` was scrubbed from every trial file (gotcha 12), so the trial-level `result.json` and `reward.txt` of this job are unusable.
    - Control: `harbor run -p tmp/smoke -a nop --ae SMOKE_FLAG=1 ... --job-name smoke-nop-ae1` -> reward 0, 42s, no redaction (non-sensitive key name).
    - Re-run with the switch in the process environment: `export CODEX_FORCE_AUTH_JSON=1; harbor run -p tmp/smoke -a codex -m openai/gpt-6-astra --ak reasoning_effort=xhigh --env docker --yes --job-name smoke-codex-2 -q` -> reward 1.0, 4m44s, 0 exceptions, `reward.txt` = `1`, `result.json` parses, zero `[REDACTED]` strings anywhere in the job dir. This is the invocation shape to use.
    - `harbor run -p tmp/smoke -a claude-code -m anthropic/claude-fable-5-1 --env-file $HOME/.harbor/claude.env --ae 'CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}' --ae CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 --ak reasoning_effort=max --env docker --yes --job-name smoke-claude -q` -> 1 exception (`UnknownApiError`: `401 OAuth access token is invalid`), reward 0, 3m19s. Diagnosis: on the host, `claude -p` with only that token set returns the same 401 and `claude -p` with the normal login returns OK, so the saved token is wrong (a partial paste is the likely cause); Harbor resolved the template and the install/run/verify path works. Token verified absent from `jobs/`.
    - Same command with the complete token (`--job-name smoke-claude-2`) -> auth passed, then `404 model_not_found` for `anthropic/claude-fable-5-1`: the shell inherited `ANTHROPIC_BASE_URL` from the Claude desktop app and Harbor keeps the provider prefix when a base URL is configured (gotcha 13). The host CLI confirms: `--model claude-fable-5-1` answers OK, `--model anthropic/claude-fable-5-1` gives the same 404.
    - `export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000; env -u ANTHROPIC_BASE_URL harbor run -p tmp/smoke -a claude-code -m anthropic/claude-fable-5-1 --env-file $HOME/.harbor/claude.env --ae 'CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}' --ae CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 --ak reasoning_effort=max --env docker --yes --job-name smoke-claude-3 -q` -> **reward 1.0, 2m35s, 0 exceptions**, `reward.txt` = `1`, zero `[REDACTED]`, token in no job file, model `claude-fable-5-1` throughout. This is the invocation shape to use.
12. `rm -rf tmp/` after the trials. `jobs/` (gitignored) keeps every trial's `result.json`, `trial.log`, agent logs and verifier output: `smoke-oracle`, `cargo-oracle`, `smoke-nop`, `smoke-nop-ae1`, `smoke-codex`, `smoke-codex-2`, `smoke-claude`, `smoke-claude-2`, `smoke-claude-3`.

## 4. `scripts/check-all.sh`

Usage: `bash scripts/check-all.sh <path-to>/tasks/<slug> [-v]`

- Mirrors `static-checks.yml` exactly: the same 28 (name, script) pairs in the
  same order, each run as `bash <checks>/<script> tasks/<slug>` with the
  working directory set to the task's repo root, so relative-path logic in the
  checks behaves as in CI. `TB_DIR` (default `../terminal-bench` next to the
  repo) points at the upstream clone that supplies `scripts/checks`.
- Exports `TASK_DIRS=tasks/<slug>`, `BASE_DIR=$TB_DIR`, and `FIX_DIRS` =
  `tasks/<slug>` when the task is new or differs from upstream in any
  non-README file, else empty. CI computes `FIX_DIRS` from the PR's changed
  files; this is the local equivalent. A task that does not exist upstream
  makes `check-task-changelog.sh` exit 0, as in CI for a new task.
- Prerequisite checks before running anything: `bash find grep sed awk comm
  python3 dockerfile-pin` on PATH; `python3 -I` is 3.11+ and imports
  `tomllib` and `yaml`. These come from the check scripts' headers
  (`check-compose-volumes.sh` needs PyYAML under `-I`, `check-image-digests.sh`
  needs `dockerfile-pin check --syntax-only --glob ...`,
  `check-cpu-threads.py` needs `tomllib`).
- Prints a `PASS/WARN/FAIL` table (WARN is nonblocking, like CI; only
  `check-cpu-threads.sh` emits `WARN` lines today) and exits non-zero on any
  FAIL. Failing checks always print their full output; `-v` prints all.

Verified: `bash scripts/check-all.sh ../terminal-bench/tasks/ctr-optimization` -> 28/28 PASS.

## 5. Job configs

Both files mirror `.github/harbor-run-defaults.yml` as resolved by
`scripts/ci/hosted_job_config.py` (same agent objects: `name`, `model_name`,
`kwargs`, `env`), with `tasks[0].path = "tasks/TASK_SLUG"` as the placeholder
to replace, `environment.type = docker`, `jobs_dir = jobs`, and
`n_concurrent_trials = 2` (a local resource choice for a 15 GB Docker VM; CI
runs hosted). The hosted-only keys `credential_mode` and per-agent `secrets`
are omitted because they are not in the local JobConfig schema.

| File | `n_attempts` | agents | extra |
|---|---|---|---|
| `scripts/run-job.json` | 3 | claude-code `anthropic/claude-fable-5-1`, kwargs `reasoning_effort=max`, env `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`; codex `openai/gpt-6-astra`, kwargs `reasoning_effort=xhigh` | |
| `scripts/cheat-job.json` | 1 | same two agents | `extra_instructions = [<verbatim docs/prompts/hack-trial-prompt.md>]` |

Auth is **not** in the configs (the CI builder rejects secret-looking env
names, and the token must never enter the repo). Supply it at run time:

```bash
export PYTHONUTF8=1
export CODEX_FORCE_AUTH_JSON=1          # process env, NOT --ae (see gotcha 12)
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000   # CI exports it in the runner env (gotcha 14)
unset ANTHROPIC_BASE_URL                # only present inside Claude Code sessions (gotcha 13)
cd $HOME/Downloads/klavis-tb3
# run (3 attempts per agent); replace TASK_SLUG in the JSON first, or override with -p
harbor run -c scripts/run-job.json --env-file $HOME/.harbor/claude.env --ae 'CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}' --yes --job-name run-1
# cheat (1 attempt per agent, hack prompt appended)
harbor run -c scripts/cheat-job.json --env-file $HOME/.harbor/claude.env --ae 'CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}' --yes --job-name cheat-1
```

Why this shape: `--env-file` loads `CLAUDE_FORCE_OAUTH=1` and the token into
the harbor process environment; the `--ae` template is resolved from that
environment (`agents/factory.py`, `resolve_env_vars`) so the token reaches the
agent as agent env and is therefore scrubbed from every trial file by
`trial.py::_scrub_jobs_dir`, while the literal value never appears on a
command line, in a config, or in `config.json` (sensitive agent env is
persisted as the `${...}` template). The two `=1` switches stay in the process
environment on purpose: their key names also match the sensitive-key regex,
and a scrubbed value of `1` destroys every `1` in `result.json` and
`reward.txt` (gotcha 12). `--ae` applies to every agent in the job; codex
ignores the Claude token. `--print-config` shows the task path as
`tasks\\TASK_SLUG` on Windows (pathlib), which is cosmetic.

## 6. Smoke trials

All on the throwaway `tmp/smoke` task (template from `harbor tasks init`,
ubuntu:24.04, shared verifier, agent timeout 600 s) unless noted. Durations are
`started_at -> finished_at` from each trial's `result.json`.

| Agent | Model (kwargs / env) | Status | Reward | Duration | Error |
|---|---|---|---|---|---|
| oracle | - | completed | 1.0 | 1m02s | none |
| oracle (upstream `cargo-flight-dispatch`, separate verifier) | - | completed | 1.0 | 1m30s (env build 25 s, verifier 52 s) | none |
| nop | - | completed | 0.0 | 35s | none |
| codex (`smoke-codex`, flag via `--ae`) | `openai/gpt-6-astra` (`reasoning_effort=xhigh`, `--ae CODEX_FORCE_AUTH_JSON=1`) | completed, but trial files corrupted by the scrubber (gotcha 12); trajectory records `gpt-6-astra` | 1.0 (console/job summary only; `reward.txt` reads `[REDACTED]`) | 5m15s | none |
| codex (`smoke-codex-2`, flag via `export`) | `openai/gpt-6-astra` (`reasoning_effort=xhigh`, `CODEX_FORCE_AUTH_JSON=1` in process env) | completed, files clean; `agent_info.version` = codex-cli 0.164.0 in the container, `model_info` = `gpt-6-astra`/openai | 1.0 | 4m44s (env 14 s, agent setup 3m12s = CLI install, agent run 20 s, verifier 29 s); 42,713 in / 174 out tokens | none |
| claude-code (`smoke-claude`) | `anthropic/claude-fable-5-1` (`reasoning_effort=max`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`, `CLAUDE_FORCE_OAUTH=1` via env file, token via `${...}` template) | pipeline completed, agent failed auth: `UnknownApiError`, Claude Code 2.1.311 in the container reported `401 OAuth access token is invalid`; `agent_info.model_info` = `claude-fable-5-1`/anthropic | 0.0 (no agent work) | 3m19s (env 14 s, agent setup 2m36s = CLI install, agent run 5 s, verifier 10 s) | 401 from the token itself: the same token gives the same 401 to the host's `claude -p`, while the host's normal claude.ai login answers fine. Re-mint and re-run (section 7). The token is absent from every file under `jobs/` and appears in `config.json` only as the `${CLAUDE_CODE_OAUTH_TOKEN}` template, so the hand-off shape is confirmed. |
| claude-code (`smoke-claude-2`, complete token) | same | pipeline completed, agent failed model lookup: `NonZeroAgentExitCodeError`, Claude Code reported `404 model_not_found` for `anthropic/claude-fable-5-1` | 0.0 | 2m42s | auth passed; the shell inherited `ANTHROPIC_BASE_URL` from the Claude desktop app, so Harbor left the `anthropic/` prefix on the model id (gotcha 13) |
| claude-code (`smoke-claude-3`, `env -u ANTHROPIC_BASE_URL`) | `anthropic/claude-fable-5-1` (`reasoning_effort=max`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000` exported and via `--ae`, `CLAUDE_FORCE_OAUTH=1` via env file, token via `${...}` template) | **completed, files clean**; Claude Code 2.1.311 in the container, trajectory and `agent_info.model_info` record `claude-fable-5-1`/anthropic | 1.0 | 2m35s (env 6 s, agent setup 1m58s = CLI install, agent run 8 s, verifier 11 s); 36,734 in / 277 out tokens | none |

## 7. claude-code: what Tanmay has to do

Harbor's claude-code agent (`agents/installed/claude_code.py`,
`_resolve_auth_env`) drops `ANTHROPIC_API_KEY` and passes
`CLAUDE_CODE_OAUTH_TOKEN` into the container when `CLAUDE_FORCE_OAUTH` is
truthy; it raises `CLAUDE_FORCE_OAUTH is set but CLAUDE_CODE_OAUTH_TOKEN is
not` otherwise. The token comes from `claude setup-token`, which is
interactive (browser login), so only Tanmay can produce it.

Once, in Git Bash:

```bash
mkdir -p ~/.harbor && printf 'CLAUDE_FORCE_OAUTH=1\nCLAUDE_CODE_OAUTH_TOKEN=%s\n' "$(claude setup-token | tail -1)" > ~/.harbor/claude.env && chmod 600 ~/.harbor/claude.env
```

(or paste the token by hand into `%USERPROFILE%\.harbor\claude.env` as those two
lines; from PowerShell, write it with `[IO.File]::WriteAllText(path, text,
[Text.Encoding]::ASCII)` so there is no BOM and no CRLF). Validate the file
before spending a trial on it, without printing the token:

```bash
( set -a; . ~/.harbor/claude.env; set +a; claude -p "Reply with the single word OK" --output-format json < /dev/null | python3 -I -c "import sys,json; j=json.loads([l for l in sys.stdin.read().splitlines() if l.startswith('{')][-1]); print('is_error:', j.get('is_error'), '| status:', j.get('api_error_status'), '| result:', str(j.get('result'))[:80])" )
```

`is_error: False | result: OK` means the token works. The first token saved on
2026-10-26 failed this check (401) and failed identically inside the trial.
Then the smoke trial is:

```bash
cd $HOME/Downloads/klavis-tb3 && export PYTHONUTF8=1 CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 && unset ANTHROPIC_BASE_URL
harbor run -p <task> -a claude-code -m anthropic/claude-fable-5-1 --env-file $HOME/.harbor/claude.env --ae 'CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}' --ae CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 --ak reasoning_effort=max --env docker --yes --job-name smoke-claude -q
```

This exact shape produced the passing `smoke-claude-3` trial (section 6).

`--env-file` is loaded into the harbor process environment with
`override=True` (`cli/jobs.py`); agents resolve env with precedence
`--ae`/config `env` first, then the process environment
(`agents/base.py::_env_sources`). `CLAUDE_FORCE_OAUTH=1` is read from the
process environment (never pass it with `--ae`, gotcha 12). The token is
handed over as a `${...}` template so it is scrubbed from the trial files
(section 5). If the trial errors with `CLAUDE_FORCE_OAUTH is set but
CLAUDE_CODE_OAUTH_TOKEN is not`, drop the token `--ae`: `--env-file` alone
also satisfies the check, at the cost of the token not being scrubbed from
`jobs/` (which is gitignored anyway). `CLAUDE_CODE_MAX_OUTPUT_TOKENS` is safe
with `--ae`: its key ends in `TOKENS`, which the sensitive-key regex excludes.
After every claude-code job, confirm `grep -rl "$(sed -n 's/^CLAUDE_CODE_OAUTH_TOKEN=//p' ~/.harbor/claude.env)" jobs/` prints nothing (it did for all three smoke jobs).

## 8. Windows-specific gotchas (read before Day 2)

1. **CRLF everywhere by default.** Global `core.autocrlf=true` turned the
   upstream clone into CRLF on checkout; `bash` tolerates it badly and a CRLF
   `test.sh` or `solve.sh` copied into a container fails on `\r`. The clone now
   has `core.autocrlf=false`; `klavis-tb3` enforces LF through
   `.gitattributes`. Files written by Windows tools can still be CRLF in the
   working tree until `git add`/re-checkout normalises them: check with
   `git ls-files --eol` or `grep -c $'\r' file`.
2. **`harbor tasks init` writes CRLF `task.toml` and `README.md`** on Windows
   (its `Dockerfile`, `test.sh`, `instruction.md`, `test_outputs.py` came out
   LF). After scaffolding the real task, run `sed -i 's/\r$//'` on those two
   (or let the `.gitattributes` normalise on commit and re-checkout).
3. **`PYTHONUTF8=1` is mandatory for harbor.** Without it the console codec is
   cp1252: `harbor tasks init` crashed printing its check mark (after writing
   files), and `--print-config` mangles the em-dashes in the cheat prompt. Put
   `export PYTHONUTF8=1` in `~/.bashrc` or run `setx PYTHONUTF8 1`.
4. **Interactive prompts abort under tool runners.** `harbor tasks init`
   asks for the organisation when stdin is not a TTY ("Organization:
   Aborted."); pass `--org terminal-bench` (the package name must be
   `terminal-bench/<slug>` for `check-task-package-name.sh`), `--author`,
   `--description`, plus `--include-canary-strings --metadata-template
   docs/task-template.toml` from the PLAN. `harbor run --yes` suppresses its
   confirmations.
5. **Paths.** `-p` accepts `tmp/smoke`, `C:/...` and `/c/...` (Git Bash
   rewrites the last for native executables). The check scripts take
   `tasks/<slug>` relative to the repo root, which `check-all.sh` arranges.
   `dockerfile-pin --glob "tasks/<slug>/**/{Dockerfile,...}"` works unchanged
   on the Windows build.
6. **Jobs dir.** `harbor run` writes to `./jobs/<job-name>/` under the current
   directory (default job name is a timestamp); run from `klavis-tb3` so it
   lands in the gitignored `jobs/`. Trial dirs hold agent logs and
   trajectories; the codex agent uploads `~/.codex/auth.json` into the
   container and claude-code receives the OAuth token as env, so never copy a
   job dir into `runs/` wholesale: trim to `result.json`, trajectories and
   verifier output and grep for `sk-`, `token`, `auth` first.
7. **`.env.local` in the cwd is auto-loaded** by `harbor run` (lowest
   precedence); `.env*` is gitignored, but still keep tokens in `~/.harbor`.
8. **Docker Desktop resources.** The VM has 16 CPUs / 15.3 GB; tasks declare
   `cpus`/`memory_mb` and Harbor enforces them (`cpu_enforcement_policy=auto`).
   With 4 GB tasks, two or three concurrent trials fit; the job configs set
   `n_concurrent_trials = 2`. Image pulls are unauthenticated Docker Hub pulls
   (rate limit 100 per 6 h per IP); `docker login` if builds start failing on pulls.
9. **Harbor removes per-trial images and containers after each trial**
   (`environment.delete` defaults to true): after all smoke runs `docker
   images` shows no `<task>__<id>__env-main` images and no dangling layers.
   The flip side is that every trial rebuilds the environment image from the
   BuildKit layer cache (the 14-25 s `environment_setup` seen here); if disk
   gets tight it is `docker builder prune`, not `docker image prune`, that
   reclaims space.
10. **`python3` is the WindowsApps alias**, which is fine for the checks'
    `python3 - <<'PYEOF'` heredocs; PyYAML is installed system-wide so `-I`
    still imports it (CI uses a venv with `PyYAML==6.0.2`).
11. **Shell heredocs with non-ASCII text fail** in this tool runner (the
    command is parsed as code past the first stray quote), and doubled
    backslashes are collapsed in transit; write such files with an editor or
    Python rather than `cat <<'EOF'`.
12. **Harbor scrubs sensitive agent env values from every trial file, and
    `1` is a value.** `trial.py::_scrub_jobs_dir` collects the values of
    agent env keys matching `KEY|SECRET|TOKEN(?!S)|PASSWORD|CREDENTIAL|AUTH`
    (`utils/env.py`) and string-replaces each with `[REDACTED]` in all text
    files under the trial dir. `--ae CODEX_FORCE_AUTH_JSON=1` therefore
    rewrote every `1` in the first codex smoke trial: `reward.txt` became
    `[REDACTED]`, `result.json` stopped parsing (`"reward": [REDACTED].0`,
    mangled timestamps and token counts), `ctrf.json` and the trajectory too.
    The job-level `result.json` and the console summary were computed before
    scrubbing and are intact. A control run (`nop` with `--ae SMOKE_FLAG=1`,
    a non-sensitive key) was untouched, confirming the key-name trigger.
    Rule: `CODEX_FORCE_AUTH_JSON=1` and `CLAUDE_FORCE_OAUTH=1` go in the
    process environment (`export`, or the `--env-file`); only real secrets go
    through `--ae`/config env, and then as `${VAR}` templates.
13. **A Claude Code session exports `ANTHROPIC_BASE_URL`, and that breaks
    the model name.** Every shell spawned by the Claude desktop app carries
    `ANTHROPIC_BASE_URL=https://api.anthropic.com` plus ~35 `CLAUDE_CODE_*`
    variables. Harbor's claude-code agent keeps the model id unstripped
    (`anthropic/claude-fable-5-1`) whenever a base URL is configured
    (`claude_code.py::_resolved_model_name`), and the CLI then fails with
    `404 model_not_found`. Run trials with `env -u ANTHROPIC_BASE_URL harbor
    run ...` from any Claude Code session (a plain PowerShell or Git Bash
    window does not have the variable). The host CLI accepts
    `claude-fable-5-1` and rejects `anthropic/claude-fable-5-1`.
14. **`CLAUDE_CODE_MAX_OUTPUT_TOKENS` has two routes into the container.**
    The claude-code agent copies it from the harbor process environment
    (`os.environ.get` in `claude_code.py::run`), which is why CI exports it
    in the runner env; `--ae` values also reach the agent's exec environment
    through `Trial`'s scoped exec env (`trial.py::scoped_exec_env`). Doing
    both, `export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000` plus the `--ae`, is
    harmless and matches CI either way.


## 9. 2026-10-28: the upstream `check-lockfiles` check

Upstream merged PR #2107 ("Check task dependency lockfiles in static CI") on
2026-10-28 at 23:21 UTC, after every trial here had run. It adds a 29th
static check, `scripts/checks/check-lockfiles.sh`, that reads each task
Dockerfile without executing it and fails any dependency install that does
not consume a committed lockfile in strict mode. Our two `pip install
name==version` lines failed it.

What was done:

1. `git -C terminal-bench pull --ff-only` -> `7b1f530`. The only CI changes
   since `6964e92` are the new check, its docs entry and a small change to
   `scripts/review/stage_task.py`; `.github/harbor-run-defaults.yml` and
   `.github/harbor-version` are unchanged.
2. Followed the pattern 53 upstream tasks use: `COPY --from=ghcr.io/astral-sh/uv@sha256:95f2...`
   (the digest 75 upstream Dockerfiles pin; it is uv 0.12.7), `COPY
   pyproject.toml uv.lock /tmp/<image>-deps/`, `UV_PROJECT_ENVIRONMENT=/usr/local
   /tmp/uv sync --locked --inexact --default-index https://pypi.org/simple`,
   then `rm -rf /tmp/uv ...`. `pyproject.toml` carries the canary comment
   (the canary check covers `.toml` under `tests/`); `uv.lock` is outside the
   check's extension list, as in every upstream task.
3. Generated each `uv.lock` inside a throwaway image (`FROM` the task's
   pinned Python image plus the pinned uv binary) with
   `docker run --rm -i <image> sh -c 'mkdir /w && cd /w && cat > pyproject.toml && uv lock -q && cat uv.lock' < pyproject.toml > uv.lock`.
   No bind mount, so no MSYS path mangling, and the file reaches disk with
   LF endings. The lock says `revision = 3`, `requires-python = "==3.12.*"`.
4. Added `"Dependency lockfiles|check-lockfiles.sh"` to the list in
   `scripts/check-all.sh` at the position `static-checks.yml` uses. Result:
   29 passed, 0 warned, 0 failed (RESULTS.md section 1).
5. Reran the oracle and nop gates on the rebuilt images (RESULTS.md section 5).

Two empty directories named `model-scanner-hardening$(echo environment` and
`model-scanner-hardening$(echo solution` had appeared under `tasks/` (a
mangled shell command from an earlier session; both empty). Removed.
