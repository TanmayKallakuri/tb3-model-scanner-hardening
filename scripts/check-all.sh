#!/usr/bin/env bash
#
# Run every Terminal-Bench 3 static check (scripts/checks/check-*.sh from the
# upstream terminal-bench clone) against one task directory, exactly the way
# .github/workflows/static-checks.yml does, and print a PASS/WARN/FAIL table.
#
# Usage:
#   scripts/check-all.sh <task-dir> [-v]
#
#   <task-dir>  path to a task directory laid out as <root>/tasks/<slug>
#   -v          also print each check's full output (failing checks always
#               print their output)
#
# Environment:
#   TB_DIR      upstream terminal-bench checkout holding scripts/checks and the
#               base tasks (default: ../terminal-bench next to this repo)
#
# Exit status is non-zero when any check FAILs. WARN lines are nonblocking,
# matching CI (only check-cpu-threads emits them today).
#
# Prerequisites, taken from the check scripts' headers:
#   - bash, find, grep, sed, awk, comm        (Git Bash on Windows is fine)
#   - python3 on PATH, 3.11+ (tomllib), importable with `python3 -I`
#   - PyYAML importable under `python3 -I`    (check-compose-volumes)
#   - dockerfile-pin v1.5.0 on PATH           (check-image-digests)
#
# FIX_DIRS (which decides whether check-task-changelog.sh demands a new Change
# Log entry) is derived locally: a task inside the upstream clone counts as a
# fix when `git status` shows changes other than its README.md; a task in
# another root counts as a fix when it differs from the upstream copy in any
# file other than its top-level README.md; a task absent upstream is new.
# CI derives it from the PR's changed-file list instead.
#
# The checks run with the working directory set to <root> and the task passed
# as the relative path tasks/<slug>, which is what CI does, so path-sensitive
# checks behave identically here and there.

set -uo pipefail

VERBOSE=0
TASK_ARG=""
for arg in "$@"; do
    case "$arg" in
        -v|--verbose) VERBOSE=1 ;;
        -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "error: unknown option: $arg" >&2; exit 2 ;;
        *)
            if [ -n "$TASK_ARG" ]; then
                echo "error: exactly one task directory expected, got '$TASK_ARG' and '$arg'" >&2
                exit 2
            fi
            TASK_ARG="$arg" ;;
    esac
done

if [ -z "$TASK_ARG" ]; then
    echo "usage: $0 <task-dir> [-v]" >&2
    exit 2
fi
if [ ! -d "$TASK_ARG" ]; then
    echo "error: not a directory: $TASK_ARG" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TB_DIR_REQUESTED=${TB_DIR:-$REPO_ROOT/../terminal-bench}
TB_DIR=$(cd "$TB_DIR_REQUESTED" 2>/dev/null && pwd) || { echo "error: TB_DIR not found: ${TB_DIR_REQUESTED}" >&2; exit 2; }
CHECKS_DIR="$TB_DIR/scripts/checks"

TASK_ABS=$(cd "$TASK_ARG" && pwd)
SLUG=$(basename "$TASK_ABS")
TASKS_PARENT=$(dirname "$TASK_ABS")
ROOT=$(dirname "$TASKS_PARENT")
if [ "$(basename "$TASKS_PARENT")" != "tasks" ]; then
    echo "error: task directory must live at <root>/tasks/<slug>, got $TASK_ABS" >&2
    exit 2
fi
REL="tasks/$SLUG"

# ---- prerequisites ---------------------------------------------------------
missing=0
for tool in bash find grep sed awk comm python3 dockerfile-pin; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool not on PATH: $tool" >&2
        missing=1
    fi
done
[ "$missing" -eq 0 ] || exit 2
if ! python3 -I -c 'import sys; assert sys.version_info >= (3, 11); import tomllib, yaml' 2>/dev/null; then
    echo "error: python3 -I must be 3.11+ with PyYAML importable (pip install PyYAML==6.0.2)" >&2
    exit 2
fi
if [ ! -d "$CHECKS_DIR" ]; then
    echo "error: checks directory not found: $CHECKS_DIR" >&2
    exit 2
fi

# ---- the CI check list, verbatim from static-checks.yml -------------------
CHECKS=(
    "Canary strings|check-canary.sh"
    "No cheat directory|check-no-cheat-dir.sh"
    "Dockerfile refs|check-dockerfile-references.sh"
    "Image digest pins|check-image-digests.sh"
    "Dependency lockfiles|check-lockfiles.sh"
    "Dockerfile sanity|check-dockerfile-sanity.sh"
    "Dockerfile platform|check-dockerfile-platform.sh"
    "Absolute paths|check-task-absolute-path.sh"
    "Test refs|check-test-file-references.sh"
    "test.sh sanity|check-test-sh-sanity.sh"
    "Task fields|check-task-fields.sh"
    "Task fix changelog|check-task-changelog.sh"
    "Task timeout cap|check-task-timeout.sh"
    "Instruction suffix|check-instruction-suffix.sh"
    "Instruction headings|check-instruction-headings.sh"
    "GPU types|check-gpu-types.sh"
    "Resource sizes|check-resource-sizes.sh"
    "Allow internet|check-allow-internet.sh"
    "No internet opt-in|check-no-allow-internet-true.sh"
    "Task slug length|check-task-slug.sh"
    "Task package name|check-task-package-name.sh"
    "Separate verifier|check-separate-verifier.sh"
    "Verifier tooling baked|check-verifier-tooling-baked.sh"
    "Trial network fetch|check-trial-network-fetch.sh"
    "Pip pinning|check-pip-pinning.sh"
    "Pytest version pin|check-pytest-version.sh"
    "Nproc usage|check-nproc.sh"
    "CPU parallelism advisory (nonblocking)|check-cpu-threads.sh"
    "Compose volumes|check-compose-volumes.sh"
)

# Environment the CI step exports. TASK_DIRS is every task dir touched by the
# PR; FIX_DIRS is the subset whose non-README files changed, and only those
# need a new Change Log entry (check-task-changelog.sh). Locally we derive
# FIX_DIRS by diffing the task against the upstream copy: a task that does not
# exist upstream is new (the check exits 0 for it), and an upstream task with
# only README changes, or no changes, is not a fix.
export TASK_DIRS="$REL"
export BASE_DIR="$TB_DIR"
task_is_fix() {
    # Returns 0 when the task counts as a "fix" of an existing upstream task.
    if [ "$ROOT" -ef "$TB_DIR" ]; then
        # The task lives inside the upstream clone: compare against HEAD, and
        # ignore only the task's own top-level README.md like CI does.
        git -C "$TB_DIR" status --porcelain --untracked-files=all -- "$REL" 2>/dev/null \
            | grep -vE "^.. ${REL}/README\.md\$" | grep -q . && return 0
        return 1
    fi
    [ -d "$BASE_DIR/$REL" ] || return 0   # new task: harmless, the check exits 0
    local changed
    changed=$(diff -rq "$BASE_DIR/$REL" "$ROOT/$REL" 2>&1 \
        | grep -vE "^Files $BASE_DIR/$REL/README\.md and $ROOT/$REL/README\.md differ\$" || true)
    [ -n "$changed" ]
}
if task_is_fix; then
    export FIX_DIRS="$REL"
else
    export FIX_DIRS=""
fi
# PR_NUMBER / REPO_URL are unset locally: changelog entries need no PR link here.

cd "$ROOT" || exit 2

echo "check-all: task=$REL root=$ROOT"
echo "check-all: checks=$CHECKS_DIR ($(git -C "$TB_DIR" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo

PASS=0; WARN=0; FAIL=0
ROWS=""
for entry in "${CHECKS[@]}"; do
    NAME="${entry%%|*}"
    SCRIPT="${entry##*|}"
    SCRIPT_PATH="$CHECKS_DIR/$SCRIPT"
    if [ ! -f "$SCRIPT_PATH" ]; then
        STATUS="FAIL"; DETAIL="script not found: $SCRIPT"
        FAIL=$((FAIL + 1))
    else
        OUT=$(bash "$SCRIPT_PATH" "$REL" 2>&1)
        RC=$?
        CLEAN=$(printf '%s\n' "$OUT" | sed -E $'s/\x1b\[[0-9;]*[A-Za-z]//g')
        WARN_LINES=$(printf '%s\n' "$CLEAN" | grep '^WARN ' || true)
        if [ "$RC" -ne 0 ]; then
            STATUS="FAIL"; FAIL=$((FAIL + 1))
            DETAIL=$(printf '%s\n' "$CLEAN" | grep '^FAIL ' | sed 's/^FAIL //' | head -3 | tr '\n' ' ')
            [ -n "$DETAIL" ] || DETAIL="(exit $RC, see output)"
        elif [ -n "$WARN_LINES" ]; then
            STATUS="WARN"; WARN=$((WARN + 1))
            DETAIL=$(printf '%s\n' "$WARN_LINES" | sed 's/^WARN //' | head -3 | tr '\n' ' ')
        else
            STATUS="PASS"; PASS=$((PASS + 1)); DETAIL=""
        fi
        if [ "$VERBOSE" -eq 1 ] || [ "$STATUS" = "FAIL" ]; then
            echo "---- $NAME ($SCRIPT) -> $STATUS (exit $RC)"
            printf '%s\n' "$CLEAN"
            echo
        fi
    fi
    ROWS+=$(printf '%-4s  %-40s  %-32s  %s' "$STATUS" "$NAME" "$SCRIPT" "$DETAIL")$'\n'
done

echo "Status  Check                                     Script                            Detail"
echo "------  ----------------------------------------  --------------------------------  ------"
printf '%s' "$ROWS"
echo
echo "check-all: $PASS passed, $WARN warned, $FAIL failed of ${#CHECKS[@]} ($REL)"
[ "$FAIL" -eq 0 ]
