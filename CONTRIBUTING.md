# Contributing

Create a branch and make focused changes. Run `studio check` before committing. Add a regression that fails without the fix.

Runtime changes need generation, intervention, and reset tests on the affected backend. Report the model, device, precision, context, and ABI. Changes to lexical rules or calibration semantics require a new rule/profile version.

Packaging changes must pass a regular wheel installation from a directory outside the checkout. Native changes must build against the pinned llama.cpp revision.

Keep generated state out of commits. Do not include model weights, session databases, user exports, or machine-specific configuration. Place small deterministic fixtures under `tests/` or `examples/`.

Documentation should describe commands, file formats, algorithm choices, and observed test results. Keep release notes in `CHANGELOG.md`.
