"""Inert CPython standard-library export metadata.

The JSON was generated from the installed standard library using inspect at
build time. Scanning only reads the JSON: it never imports a module named by an
artifact. Metadata records aliases, classes, and signatures, not call effects.
Effects and all operations remain explicitly implemented in capabilities.py.
"""
import json
import os

with open(os.path.join(os.path.dirname(__file__), 'stdlib_metadata.json'), encoding='utf-8') as _file:
    _data = json.load(_file)
NAMESPACES = _data['namespaces']
SYMBOLS = _data['symbols']
del _data
