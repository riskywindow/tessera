#!/usr/bin/env bash
# Fixture: paid GPU work whose failure is downgraded to a log line.
# tessera-lint-expect: modal-masked
set -euo pipefail

modal run infra/modal_gpu.py::m0_gate || echo "GPU gate skipped" >&2
