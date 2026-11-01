"""Report objects and JSON serialisation (see /app/docs/report-schema.md)."""
from __future__ import annotations

import json

KINDS = ("code-execution", "process", "file-access", "network", "unsupported")


def _clean(text, limit=2000) -> str:
    """Free text derived from an artifact: no lone surrogates, bounded length."""
    text = str(text)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text.encode("utf-8", "replace").decode("utf-8")


class Finding:
    __slots__ = ("kind", "location", "detail")

    def __init__(self, kind: str, location: str, detail: str):
        if kind not in KINDS:
            raise ValueError(f"unknown finding kind {kind!r}")
        self.kind = kind
        self.location = _clean(location)
        self.detail = _clean(detail)

    def __repr__(self):
        return f"Finding({self.kind!r}, {self.location!r}, {self.detail!r})"

    def __eq__(self, other):
        return isinstance(other, Finding) and (self.kind, self.location, self.detail) == \
            (other.kind, other.location, other.detail)

    __hash__ = None


class Result:
    __slots__ = ("path", "findings")

    def __init__(self, path: str, findings=None):
        self.path = path
        self.findings = list(findings or [])

    @property
    def verdict(self) -> str:
        return "malicious" if self.findings else "clean"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "verdict": self.verdict,
            "findings": [{"kind": f.kind, "location": f.location, "detail": f.detail} for f in self.findings],
        }


def to_json(results) -> str:
    # ASCII escapes keep the document writable whatever the path argument holds
    return json.dumps({"scanner": "modelgate", "results": [r.to_dict() for r in results]}, indent=2) + "\n"
