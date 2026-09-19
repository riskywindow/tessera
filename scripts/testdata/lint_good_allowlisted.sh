#!/usr/bin/env bash
# Fixture: the exceptions the allowlist exists to make visible rather than
# silent. Each one names the rule and says why.
set -euo pipefail

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

# Best-effort cleanup of a path another agent may already have reaped; the
# only information a failure carries is that it is already gone.
rm -rf "/tmp/tessera-build/scratch-$$" || true  # tessera-lint: allow-or-true path may already be gone

# ccache is optional on this host, and its absence is not a build failure.
command -v ccache >/dev/null || :  # tessera-lint: allow-or-true ccache is optional here

echo "scratch at $scratch"
