#!/usr/bin/env python3
"""Stage the implementation-rubric review the way upstream CI does, for a
local task directory instead of a GitHub commit.

Mirrors terminal-bench/scripts/review/stage_task.py: a Harbor task whose
environment holds the task under review at /app/task-under-review/<name>,
the rubric at /app/rubric.toml and the word-count helper, whose instruction
is the reviewer template plus the rubric, and whose verifier only checks
that /app/verdicts.json was written.  Run the staged task with the
claude-code agent on the subscription (see RESULTS.md section 4).

    python3 scripts/rubric-review/stage.py tasks/model-scanner-hardening tmp/rubric-review
"""
import argparse
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
TB = REPO.parent / "terminal-bench"


def stage(task_dir: pathlib.Path, destination: pathlib.Path) -> None:
    task_dir = task_dir.resolve()
    name = task_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    environment = destination / "environment"
    tests = destination / "tests"
    environment.mkdir(parents=True)
    tests.mkdir()
    shutil.copytree(task_dir, environment / "task-under-review" / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".session-state"))
    shutil.copy(TB / "scripts/review/instruction_word_counts.py", environment / "instruction_word_counts.py")
    shutil.copy(TB / "docs/prompts/task-implementation.toml", environment / "rubric.toml")
    (environment / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "ca-certificates curl git nodejs npm procps python3 "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "COPY task-under-review /app/task-under-review\n"
        "COPY instruction_word_counts.py /app/instruction_word_counts.py\n"
        "COPY rubric.toml /app/rubric.toml\n"
        "WORKDIR /app\n",
        encoding="utf-8",
    )
    # Upstream appends the full rubric text to the instruction as well. On
    # Windows, Harbor passes the instruction on the `docker compose exec`
    # command line and a 65 KB instruction exceeds the CreateProcess limit
    # (WinError 206), so here the agent reads the identical rubric from
    # /app/rubric.toml, which the template instruction already points at.
    instruction = (TB / "scripts/rubric-regression/templates/instruction.md").read_text(encoding="utf-8")
    (destination / "instruction.md").write_text(instruction + "\n\nThe rubric is the file `/app/rubric.toml`; read it in full before writing the verdicts.\n", encoding="utf-8")
    (destination / "task.toml").write_text(
        'schema_version = "1.0"\n'
        'artifacts = ["/app/verdicts.json"]\n\n'
        "[agent]\n"
        "timeout_sec = 1800.0\n\n"
        "[environment]\n"
        "build_timeout_sec = 900.0\n"
        "cpus = 2\n"
        "memory_mb = 4096\n",
        encoding="utf-8",
    )
    (tests / "test.sh").write_text(
        "#!/bin/bash\n"
        "if [ -f /app/verdicts.json ] && [ -s /app/verdicts.json ]; then\n"
        "  echo 1 > /logs/verifier/reward.txt\n"
        "else\n"
        "  echo 0 > /logs/verifier/reward.txt\n"
        "fi\n",
        encoding="utf-8",
    )
    print("staged review task at", destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    stage(args.task_dir, args.output)


if __name__ == "__main__":
    main()
