#!/usr/bin/env bash
# Fixture: no errexit, so every command below is advisory.
# tessera-lint-expect: missing-set-e

cmake --preset gcc-debug
cmake --build --preset gcc-debug -j3
ctest --preset gcc-debug
