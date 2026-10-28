#!/usr/bin/env bash
# Launch the /run and /cheat equivalents for the task, with the exact
# invocation shape BRINGUP.md sections 5 and 7 establish on this machine.
#
#   scripts/run-trials.sh run    <job-name>   # 3 attempts per agent, scripts/run-job.json
#   scripts/run-trials.sh cheat  <job-name>   # 1 attempt per agent with the hack prompt, scripts/cheat-job.json
#   scripts/run-trials.sh run-claude|run-codex <job-name>   # per-agent reruns after infrastructure errors
#
# Auth is supplied at run time only: the Claude OAuth token comes from
# ~/.harbor/claude.env (CLAUDE_FORCE_OAUTH=1 plus the token) and is handed to
# the agent as a ${...} template so Harbor scrubs it from every trial file;
# codex uses ~/.codex/auth.json via CODEX_FORCE_AUTH_JSON=1 in the process
# environment (never --ae: a scrubbed value of "1" corrupts the trial files).
set -euo pipefail
kind="${1:?run|cheat}"
name="${2:?job name}"
cd "$(dirname "$0")/.."
case "$kind" in
  run) config=scripts/run-job.json ;;
  run-claude) config=scripts/run-job-claude.json ;;   # reruns of claude-code attempts that ended in an infra error
  run-codex) config=scripts/run-job-codex.json ;;     # reruns of codex attempts that ended in an infra error
  cheat) config=scripts/cheat-job.json ;;
  *) echo "unknown kind: $kind" >&2; exit 2 ;;
esac
export PYTHONUTF8=1
export CODEX_FORCE_AUTH_JSON=1
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
unset ANTHROPIC_BASE_URL || true
mkdir -p tmp/harbor-logs
echo "launching $kind trials as job $name (config $config)"
harbor run -c "$config" --env-file "$HOME/.harbor/claude.env" --ae 'CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}' --env docker --yes --job-name "$name"
