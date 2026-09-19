# Lint fixtures

These files are inputs to `scripts/lint_exit_codes.sh --self-test`, not working
code. Every `lint_bad_*` file is deliberately broken and declares the rules it
must trigger:

    # tessera-lint-expect: <rule> [<rule> ...]

The self-test fails unless the scanner reports *exactly* that set of rules, so
a bad fixture cannot pass by being rejected for the wrong reason. Every
`lint_good_*` file must produce no findings at all.

The whole directory is excluded from the normal scan (`is_fixture` in
`scripts/lint_exit_codes.sh`); nothing here is ever executed.
