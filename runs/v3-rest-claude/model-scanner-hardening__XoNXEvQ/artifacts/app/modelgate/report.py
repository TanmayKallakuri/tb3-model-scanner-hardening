"""Report objects and JSON serialisation (see /app/docs/report-schema.md)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

KINDS = ("code-execution", "process", "file-access", "network", "unsupported")


@dataclass
class Finding:
    kind: str
    location: str
    detail: str

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown finding kind {self.kind!r}")


@dataclass
class Result:
    path: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "malicious" if self.findings else "clean"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "verdict": self.verdict,
            "findings": [{"kind": f.kind, "location": f.location, "detail": f.detail} for f in self.findings],
        }


def to_json(results: list[Result]) -> str:
    return json.dumps({"scanner": "modelgate", "results": [r.to_dict() for r in results]}, indent=2, ensure_ascii=True) + "\n"
