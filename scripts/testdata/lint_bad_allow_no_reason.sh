#!/usr/bin/env bash
# Fixture: an allow directive with no justification suppresses nothing.
# tessera-lint-expect: or-true bad-allow-directive
set -euo pipefail

rm -f /tmp/tessera-stale.lock || true  # tessera-lint: allow-or-true
