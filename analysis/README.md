# analysis/

Why each agent trial failed the verifier and why each cheat trial got
nothing, written from the trajectories and verifier reports under `runs/`.

## Version 1.1.0 (the shipped version)

- [`v3-failures.md`](v3-failures.md): the five genuine failures on `gpt-6.1-sol` and `opus-5-5`, what each agent built, where each scanner was wrong, and the one corpus family to watch. The CI analysis step (`harbor analyze` with upstream's `trial-analysis.toml`) also ran on every completed trial; its verdicts are in RESULTS.md section 0.5 and its outputs under `runs/v3-analyze2-*`.

## Version 1.0.0

- [`v2-hardening.md`](v2-hardening.md): why and how the corpus was rebuilt after codex passed v1; the probe method and its numbers. Version 1.1.0 later removed the items that came only from the probe's third round (RESULTS.md 0.1).
- [`run-trials.md`](run-trials.md): the `/run` trials on the CI default pair: Part 1 on the v1 corpus, Part 2 on v2.
- [`cheat-trials.md`](cheat-trials.md): the local red-team pass and the `/cheat` trials.
- `trials-summary.md`: machine-extracted per-trial summary (`scripts/summarize-trial.py`).
- `probe/`: raw outputs of the probe rounds and of the five v1 agent scanners on the v2 corpus.
