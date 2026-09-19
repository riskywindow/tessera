#!/usr/bin/env python3
"""Fixture: the handler that eats the error."""
# tessera-lint-expect: except-pass
import json
import pathlib


def load(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError:
        pass
    return {}
