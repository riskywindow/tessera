#!/usr/bin/env bash
# Fixture: paid GPU work whose failure stays a failure.
set -euo pipefail

modal run infra/modal_gpu.py::m0_gate
echo "gate finished"
