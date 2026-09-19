#!/usr/bin/env python3
"""Fixture: the ways a Python runner is allowed to call out to a subprocess."""
import subprocess
import sys


def build(preset: str) -> None:
    subprocess.run(["cmake", "--build", "--preset", preset, "-j3"], check=True)


def test(preset: str) -> int:
    proc = subprocess.run(
        ["ctest", "--preset", preset],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
    return proc.returncode


def main() -> int:
    build("gcc-debug")
    try:
        return test("gcc-debug")
    except FileNotFoundError as exc:
        raise SystemExit(f"ctest is not installed: {exc}") from exc


if __name__ == "__main__":
    sys.exit(main())
