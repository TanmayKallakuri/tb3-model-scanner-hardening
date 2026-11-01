# scripts/

Everything runs from the repository root. Where a script needs the pinned
Python environment it says so in its docstring (`mgdev`: Python 3.12 with the
task's pinned packages; `gtval`: the same plus the real ML frameworks).

## Version 1.1.0 (the shipped version, RESULTS.md section 0)

| File | Purpose |
|---|---|
| `run-v3.sh` | Launches the `/run` and `/cheat` trials on the brief's models: `pilot-codex`, `pilot-claude`, `rest-codex`, `rest-claude`, `replace-codex`, `cheat-codex`, `cheat-claude`, and `api-replace-codex`, `api-cheat-codex` for an OpenAI API key. Refuses to reuse a job name. |
| `run-job-codex-sol.json`, `run-job-codex-sol-rest.json`, `run-job-codex-sol-api.json` | codex `openai/gpt-6.1-sol`, `reasoning_effort=xhigh`: one attempt, two attempts, one attempt on an API key. |
| `run-job-claude-opus55.json`, `run-job-claude-opus55-rest.json` | claude-code `anthropic/claude-opus-5-5`, `reasoning_effort=max`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`: one attempt, two attempts. |
| `cheat-job-codex.json`, `cheat-job-codex-api.json`, `cheat-job-claude-opus55.json` | The same agents with upstream's `docs/prompts/hack-trial-prompt.md` (commit `4031b48`) as the extra instruction. |
| `probe/trim_matrix.py` | Regrades every saved agent scanner on the 1.0.0 corpus (648 items) and the 1.1.0 corpus (564), the kill check run before any 1.1.0 trial. |
| `v3-table.py` | Prints the 1.1.0 trial table from the trimmed job copies under `runs/v3-*`. |

## Gates and tooling (both versions)

| File | Purpose |
|---|---|
| `check-all.sh` | Runs upstream's 29 static checks against the task the way `static-checks.yml` does (`TB_DIR` points at the upstream clone). |
| `rubric-review/stage.py` | Stages the 37-criterion implementation rubric review as a Harbor task, the way upstream's `stage_task.py` does. |
| `gt-validate/validate.py` | Labels every hidden item by loading it with the real framework loaders under armed canaries and writes `tests/manifest.json`; `report.json` is its last run. |
| `shortcut-audit/audit.py` | Shows that cheap classifiers (names, sizes, entropy, extensions) score reward 0 on the hidden corpus. |
| `build-samples.py` | Writes the visible sample set under `/app/samples` and its hash list for the verifier. |
| `gen-tests/`, `oracle-tests/`, `harness-tests/` | pytest suites for the corpus generator, the reference scanner and the verifier harness. |
| `probe/probe.py`, `probe/corpus_matrix.py` | The v2 hardening probe: candidate constructs run against every scanner, and the full-corpus matrix (`analysis/v2-hardening.md`). |
| `trim-run.py` | Copies a Harbor job into `runs/<job>` without secrets, session state or local paths. |
| `summarize-trial.py` | Per-trial summary of a job for the write-up. |

## Version 1.0.0 (RESULTS.md sections 1 to 7)

| File | Purpose |
|---|---|
| `run-trials.sh` | The 1.0.0 launcher (`run`, `cheat`, `run-claude`, `run-codex`). |
| `run-job.json`, `run-job-claude.json`, `run-job-claude-1.json`, `run-job-codex.json`, `run-job-codex-1.json`, `run-job-codex-2.json`, `cheat-job.json`, `cheat-job-claude.json` | Job configs on the CI default pair of the time (`anthropic/claude-fable-5-1`, `openai/gpt-6-astra`), kept as the record of how those trials were launched. |
