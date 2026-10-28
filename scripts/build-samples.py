#!/usr/bin/env python3
"""Write the visible sample corpus (/app/samples in the agent image) and the
verifier's list of sample hashes.

    MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/repo" mgdev:latest \
        python /repo/scripts/build-samples.py

The samples come from the same generator as the hidden corpus but from a
different seed (gen.corpus.SAMPLE_SEED) and are a deterministic subset of 15
malicious and 15 clean items covering the families.  The script asserts that
no sample shares a sha256 with any hidden item.
"""
import hashlib
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path("/repo")
TASK = REPO / "tasks" / "model-scanner-hardening"
sys.path.insert(0, str(TASK / "tests"))

from gen import corpus  # noqa: E402

OUT = TASK / "environment" / "app" / "samples"
HASHES = TASK / "tests" / "samples_hashes.json"


def main():
    samples = corpus.generate_samples()
    hidden = {corpus.item_sha256(i) for i in corpus.generate()}
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    labels = {}
    hashes = []
    for k, item in enumerate(samples):
        h = corpus.item_sha256(item)
        assert h not in hidden, f"sample {item.family} collides with the hidden corpus"
        hashes.append(h)
        name = f"sample_{k + 1:02d}{item.ext}"
        if item.ext == "":
            d = OUT / name
            for rel, data in item.files.items():
                p = d / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        else:
            (OUT / name).write_bytes(item.files["artifact"])
        labels[name] = item.label
    (OUT / "labels.json").write_text(json.dumps(dict(sorted(labels.items())), indent=2) + "\n", encoding="utf-8")
    HASHES.write_text(json.dumps(sorted(hashes)) + "\n", encoding="utf-8")
    counts = {"malicious": sum(v == "malicious" for v in labels.values()), "clean": sum(v == "clean" for v in labels.values())}
    print("samples:", counts, "families:", sorted({i.family for i in samples}))
    print("wrote", OUT, "and", HASHES)


if __name__ == "__main__":
    main()
