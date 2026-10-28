# Failure analysis

Why each agent trial failed the verifier, and why each cheat trial got
nothing. Written from the trajectories and verifier reports under `runs/`
(the hosted CI's analysis step runs `harbor analyze` with
`docs/prompts/trial-analysis.toml` over the same material; it needs an API
key, so the analysis here is hand-written from the same inputs and follows
the same questions: what the agent did, where it stopped, what the verifier
measured, and whether the failure is a genuine model failure or an
infrastructure error).

- [`v2-hardening.md`](v2-hardening.md): why and how the corpus was rebuilt after codex passed v1; the probe method and its numbers.
- [`run-trials.md`](run-trials.md): the official `/run` trials (3 attempts per agent; on v2 codex completed one, the rest ended on the provider's limits): Part 1 on the v1 corpus, Part 2 on v2.
- `probe/`: raw outputs of the probe rounds and of the five v1 agent scanners on the v2 corpus (`scripts/probe/`).
- [`cheat-trials.md`](cheat-trials.md): the local red-team pass and the official `/cheat` trials.
- `trials-summary.md`: machine-extracted per-trial summary (`scripts/summarize-trial.py`).
