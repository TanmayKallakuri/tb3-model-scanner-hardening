"""ModelGate: the registry's static scanner for uploaded model artifacts.

Entry point: ``python3 -I /app/modelgate/cli.py scan --output <report.json> <path>``.
The report contract is documented in /app/docs/report-schema.md and the
policy the verdicts must match in /app/docs/threat-model.md.
"""

__version__ = "0.4.2"
