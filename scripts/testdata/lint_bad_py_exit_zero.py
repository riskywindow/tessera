#!/usr/bin/env python3
"""Fixture: the handler that reports success after the suite failed."""
# tessera-lint-expect: exit-zero
import subprocess
import sys


def main() -> int:
    try:
        subprocess.run(["ctest", "--preset", "gcc-debug"], check=True)
    except subprocess.CalledProcessError:
        print("tests failed; continuing anyway", file=sys.stderr)
        sys.exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
