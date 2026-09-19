#!/usr/bin/env bash
# Fixture: a failure thrown away by a trailing no-op.
# tessera-lint-expect: or-true
set -euo pipefail

rm -rf /tmp/tessera-does-not-exist || true
