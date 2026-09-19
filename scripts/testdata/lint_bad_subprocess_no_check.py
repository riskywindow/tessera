#!/usr/bin/env python3
"""Fixture: a subprocess whose exit status nobody ever reads."""
# tessera-lint-expect: subprocess-no-check
import subprocess


def build(preset: str) -> None:
    subprocess.run(
        ["cmake", "--build", "--preset", preset, "-j3"],
        cwd="/root/tessera",
    )
