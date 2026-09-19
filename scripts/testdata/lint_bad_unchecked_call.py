#!/usr/bin/env python3
"""Fixture: an API that hands back a status nothing forces you to read."""
# tessera-lint-expect: unchecked-call
import os


def clean() -> None:
    os.system("rm -rf /tmp/tessera-build/scratch")
