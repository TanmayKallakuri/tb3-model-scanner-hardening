# runs/

Trimmed copies of every Harbor job cited in RESULTS.md, rendered by
`scripts/trim-run.py`: each trial's `result.json`, `trial.log`, verifier
output (`reward.txt`, `reward-details.json`, `test-stdout.txt`), the
collected `/app/modelgate/` the agent left behind, and the agent's console
log and `trajectory.json`. Dropped: `config.json` (agent environment),
Claude Code session state, and anything that could hold a token or a local
path (the home directory appears as `<home>`).

The collected scanners under `*/artifacts/app/modelgate/` are the agents'
own code, kept unmodified as evidence; they are not part of the task.

## Version 1.1.0 (RESULTS.md section 0)

| Directory | What it is |
|---|---|
| `v3-static-checks/` | Static-check transcripts, 29 of 29 (`check-all-1.1.0.log`, and `-final` after the last edits) |
| `v3-gt-validate/` | Ground truth by execution, 564 of 564 valid |
| `v3-trim-matrix/` | The 24 saved scanners regraded on the 1.0.0 and 1.1.0 corpora (step 0) |
| `v3-gate-oracle-2/`, `v3-gate-nop-2/` | Oracle 1.0, nop 0.0 on the final files |
| `v3-rubric-review-2/` | Implementation rubric, 36 pass, 0 fail, 1 not applicable |
| `v3-pilot-codex/`, `v3-rest-codex/` | codex `gpt-6.1-sol` `/run`: two genuine failures (`pZmdSZW`, `ZcCeMyV`) and one attempt cut by the ChatGPT usage limit (`yxw2uDi`) |
| `v3-replace-codex/`, `v3-api-replace-codex/` | Two further launches of the third codex attempt, cut by the ChatGPT limit and by API credit; not counted |
| `v3-pilot-claude/`, `v3-rest-claude/` | claude-code `opus-5-5` `/run`: three genuine failures (`PiWwP3W`, `XoNXEvQ`, `qAQpeTC`) |
| `v3-cheat-claude/` | claude-code `/cheat`: a genuine attempt, no bypass, reward 0 |
| `v3-cheat-codex/`, `v3-api-cheat-codex/` | codex `/cheat`: the first cut by the usage limit, the second refused by the provider's safeguard; reward 0 |
| `v3-analyze2-*/`, `v3-analyze-prompts/` | `harbor analyze` on every completed trial, with the upstream prompts used |

## Version 1.0.0 (RESULTS.md sections 1 to 7)

| Directory | What it is |
|---|---|
| `gate-oracle*/`, `gate-nop*/` | Oracle and nop on the v1, v2 and lockfile-rebuilt images |
| `rubric-review-2/`, `-3/`, `-4/` | Rubric reviews of v1, v2, and v2 with the README's experience section |
| `redteam-1/` | The local red-team pass run before the official trials |
| `run-1/` to `run-11/` | The `/run` trials on the CI default pair: v1 (`run-1` to `run-3`) and v2 (`run-4` onward), including the launches that ended on usage limits |
| `cheat-1-claude/`, `cheat-1-codex/` | The `/cheat` trials |
