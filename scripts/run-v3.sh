#!/usr/bin/env bash
# Launch the 1.1.0 trials on the models the Klavis brief names:
# codex openai/gpt-6.1-sol (xhigh) and claude-code anthropic/claude-opus-5-5 (max).
#
#   scripts/run-v3.sh pilot-codex   # 1 /run attempt, scripts/run-job-codex-sol.json
#   scripts/run-v3.sh pilot-claude  # 1 /run attempt, scripts/run-job-claude-opus55.json
#   scripts/run-v3.sh rest-codex    # 2 more /run attempts, scripts/run-job-codex-sol-rest.json
#   scripts/run-v3.sh rest-claude   # 2 more /run attempts, scripts/run-job-claude-opus55-rest.json
#   scripts/run-v3.sh cheat-codex   # 1 /cheat attempt, scripts/cheat-job-codex.json
#   scripts/run-v3.sh cheat-claude  # 1 /cheat attempt, scripts/cheat-job-claude-opus55.json
#   scripts/run-v3.sh replace-codex|replace-claude  # 1 /run attempt replacing one that ended in an infra error
#
# Auth as in scripts/run-trials.sh: the Claude OAuth token from
# ~/.harbor/claude.env as a ${...} template, so Harbor scrubs only the token;
# codex through ~/.codex/auth.json with CODEX_FORCE_AUTH_JSON=1 in the process
# environment (never --ae: a scrubbed value of "1" corrupts the trial files).
# The job name is v3-<mode>; an existing job directory is never reused.
set -euo pipefail
mode="${1:?pilot-codex|pilot-claude|rest-codex|rest-claude|cheat-codex|cheat-claude}"
cd "$(dirname "$0")/.."
case "$mode" in
  pilot-codex) config=scripts/run-job-codex-sol.json ;;
  pilot-claude) config=scripts/run-job-claude-opus55.json ;;
  rest-codex) config=scripts/run-job-codex-sol-rest.json ;;
  rest-claude) config=scripts/run-job-claude-opus55-rest.json ;;
  replace-codex) config=scripts/run-job-codex-sol.json ;;   # replaces a codex attempt cut by the usage window
  replace-claude) config=scripts/run-job-claude-opus55.json ;;
  api-replace-codex) config=scripts/run-job-codex-sol-api.json ;;   # codex on an OpenAI API key
  api-cheat-codex) config=scripts/cheat-job-codex-api.json ;;
  cheat-codex) config=scripts/cheat-job-codex.json ;;
  cheat-claude) config=scripts/cheat-job-claude-opus55.json ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
name="v3-$mode"
if [ -e "jobs/$name" ]; then
  echo "jobs/$name exists; refusing to reuse it" >&2
  exit 3
fi
export PYTHONUTF8=1
export CODEX_FORCE_AUTH_JSON=1
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
unset ANTHROPIC_BASE_URL || true
mkdir -p tmp/harbor-logs
echo "launching $mode as job $name (config $config)"
# The Claude token goes only to claude-code jobs, never into a codex container.
# api-* modes use the key in ~/.harbor/codex.env (OPENAI_API_KEY=...) as a
# ${...} template, so Harbor scrubs it from every trial file, and leave
# CODEX_FORCE_AUTH_JSON unset so codex authenticates with the key.
auth=()
case "$mode" in
  *-claude) auth=(--env-file "$HOME/.harbor/claude.env" --ae 'CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}') ;;
  api-*)
    [ -f "$HOME/.harbor/codex.env" ] || { echo "missing ~/.harbor/codex.env (one line: OPENAI_API_KEY=...)" >&2; exit 4; }
    unset CODEX_FORCE_AUTH_JSON
    auth=(--env-file "$HOME/.harbor/codex.env" --ae 'OPENAI_API_KEY=${OPENAI_API_KEY}') ;;
esac
harbor run -c "$config" --agent-setup-timeout-multiplier 3 ${auth[@]+"${auth[@]}"} --env docker --yes --job-name "$name"
