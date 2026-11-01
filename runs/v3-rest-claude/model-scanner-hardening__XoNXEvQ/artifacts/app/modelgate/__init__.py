"""ModelGate: the registry's static scanner for uploaded model artifacts.

Entry point: ``python3 -I /app/modelgate/cli.py scan --output <report.json> <path>``.
The report contract is documented in /app/docs/report-schema.md and the
policy the verdicts must match in /app/docs/threat-model.md.

How it works (nothing from an artifact is ever imported, called or written):

* ``formats``      decides the format of every file and archive member by
                   content and applies the semantics of its standard loader
                   (pickle.load until EOF, joblib.load, torch.load zip/legacy,
                   numpy.load, Keras, SavedModel, archives).
* ``unpickler``    re-implements CPython's unpickler (the C one used by
                   pickle.load and the pure-Python one used by joblib.load)
                   over symbolic values (``values``).
* ``semantics``    decides what each attribute lookup, call and iteration
                   would do, using ``pyindex`` (an offline index of the
                   standard library and numpy: module attributes, aliases,
                   class MROs, signatures) so that ``posix.system`` is
                   ``os.system`` and ``shutil.os`` is ``os``.
* ``handlers``     models the callables that matter: routes to other
                   callables (getattr, partial, map consumed by list, threads,
                   ...), nested loaders (pickle.loads of decoded data, ...) and
                   pure data transformations (base64, zlib, string building).
* ``caps``         the capability-bearing callables, letter A-D.
* ``keras_scan``, ``savedmodel_scan``, ``h5attrs`` and ``containers`` handle
  the other formats; ``h5attrs`` reads HDF5 root attributes in pure Python
  because the HDF5 library can hang on malformed files.

The index (``data/pyindex.json``) is generated offline from this Python
installation; regenerate it when the interpreter or numpy changes.
"""

__version__ = "0.5.0"
