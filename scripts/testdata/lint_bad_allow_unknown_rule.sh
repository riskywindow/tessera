#!/usr/bin/env bash
# Fixture: an allow directive naming a rule that does not exist. A typo in a
# rule name must not silently become a blanket suppression.
# tessera-lint-expect: bad-allow-directive
set -euo pipefail

# tessera-lint: allow-or-true-ish this rule name is a typo
echo "nothing to see here"
