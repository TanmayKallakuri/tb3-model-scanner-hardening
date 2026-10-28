# ModelGate scanner contract

## Invocation

The registry runs the scanner once per uploaded artifact, from a working
directory that is not `/app`, with no environment variables beyond `PATH`,
`HOME` and `TMPDIR`:

```
python3 -I /app/modelgate/cli.py scan --output <report.json> <path>
```

`<path>` is the artifact: a file, or a directory for a TensorFlow SavedModel.
The scanner must write exactly one JSON document to `<report.json>` and exit
with status 0 once it is written. Exit status 2 is reserved for a usage error
(missing or unknown arguments). Only `/app/modelgate/` is deployed to the
registry; the scanner may use nothing outside it except the Python packages
already installed in this environment. The registry's held-out corpus holds
roughly 650 artifacts and is scanned in one pass, one invocation per artifact:
the whole pass must finish within 900 seconds and no single artifact may take
more than 60 seconds.

## Report

```json
{
  "scanner": "modelgate",
  "results": [
    {
      "path": "<the path argument exactly as given>",
      "verdict": "malicious",
      "findings": [
        {
          "kind": "code-execution",
          "location": "archive/data.pkl",
          "detail": "REDUCE builtins.eval"
        }
      ]
    }
  ]
}
```

Rules:

- `results` has exactly one entry per path argument, in argument order, and
  `path` repeats the argument byte for byte.
- `verdict` is `"malicious"` or `"clean"`.
- `findings` is a list that is non-empty if and only if the verdict is
  `"malicious"`.
- each finding is an object with exactly the three string keys `kind`,
  `location` and `detail`.
- `kind` is one of `"code-execution"`, `"process"`, `"file-access"`,
  `"network"`, `"unsupported"`.
- `location` names where the finding is (a container member, a pickle stream
  and opcode offset, a graph node, a layer path) and `detail` says what it is.
  Both are free text for humans.

An artifact the scanner cannot analyse must still receive a verdict. The
registry policy is fail-closed: report it `"malicious"` with one finding of
kind `"unsupported"` that says why.

A report that is missing, is not valid JSON, violates any rule above, or is
not written within the time limit counts as a wrong verdict for that artifact.
