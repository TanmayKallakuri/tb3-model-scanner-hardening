#!/usr/bin/env python3
"""Print the 1.1.0 trial table (markdown) from the trimmed job copies under runs/v3-*.

One row per trial: agent, model, reward, exception, agent minutes, tokens,
misses and false positives with their families, and whether every assistant
message stayed on the configured model (claude-code only).

    python3 scripts/v3-table.py
"""
import collections
import datetime
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
JOBS = ["v3-pilot-codex", "v3-rest-codex", "v3-replace-codex", "v3-api-replace-codex", "v3-pilot-claude",
        "v3-rest-claude", "v3-cheat-codex", "v3-api-cheat-codex", "v3-cheat-claude"]


def minutes(phase):
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return None
    a, b = (datetime.datetime.fromisoformat(phase[k].replace("Z", "+00:00")) for k in ("started_at", "finished_at"))
    return round((b - a).total_seconds() / 60, 1)


def rows():
    for job in JOBS:
        root = REPO / "runs" / job
        for trial in sorted(p for p in root.glob("*/") if (p / "result.json").exists()):
            r = json.loads((trial / "result.json").read_text(encoding="utf-8"))
            cfg = r.get("config", {}).get("agent", {})
            ar = r.get("agent_result") or {}
            ei = r.get("exception_info") or {}
            details = trial / "verifier" / "reward-details.json"
            fn = fp = []
            if details.exists():
                d = json.loads(details.read_text(encoding="utf-8"))
                wrong = [i for i in d["items"] if not i["correct"]]
                fn = [i["family"] for i in wrong if i["label"] == "malicious"]
                fp = [i["family"] for i in wrong if i["label"] == "clean"]
            models = ""
            log = trial / "agent" / "claude-code.txt"
            if log.exists():
                text = log.read_text(encoding="utf-8", errors="replace")
                c = collections.Counter(re.findall(r'"model":"([^"]+)"', text))
                fallbacks = text.count('"type":"fallback"') + text.count("model_refusal_fallback")
                models = ", ".join(f"{m} x{n}" for m, n in c.items()) + f"; fallbacks {fallbacks}"
            reward = (trial / "verifier" / "reward.txt").read_text().strip() if (trial / "verifier" / "reward.txt").exists() else "-"
            yield {
                "job": job, "trial": trial.name.split("__")[-1], "agent": cfg.get("name"), "model": cfg.get("model_name"),
                "reward": reward, "exception": ei.get("exception_type") or "none", "minutes": minutes(r.get("agent_execution")),
                "tokens_in": ar.get("n_input_tokens"), "tokens_out": ar.get("n_output_tokens"),
                "fn": fn, "fp": fp, "models": models,
            }


def main():
    print("| Job / trial | Agent, model | Reward | Exception | Agent min | Tokens in / out | Misses | False positives |")
    print("|---|---|---|---|---|---|---|---|")
    for x in rows():
        tok = f"{x['tokens_in'] / 1e6:.1f}M / {x['tokens_out'] / 1e3:.0f}k" if x["tokens_in"] else "-"
        print(f"| `{x['job']}` / `{x['trial']}` | {x['agent']}, `{x['model']}` | **{x['reward']}** | {x['exception']} | {x['minutes']} | {tok} | {len(x['fn'])} | {len(x['fp'])} |")
    print()
    for x in rows():
        print(f"- `{x['trial']}` ({x['agent']}): misses {', '.join(f'`{f}`' for f in x['fn']) or 'none'}; "
              f"false positives {', '.join(f'`{f}`' for f in sorted(set(x['fp']))) or 'none'}"
              + (f"; models: {x['models']}" if x["models"] else ""))


if __name__ == "__main__":
    main()
