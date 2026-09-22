# Changelog

## 0.3.0

- Added CUDA BF16 and NF4 persona inference with Auto/CPU/CUDA selection.
- Added device-specific memory admission and separate numerical identities.
- Corrected source-seal startup errors and validated seals before worker launch.
- Fixed benchmark loader return values, sampling controls, and cache identity.
- Packaged configuration, documentation, paper data, native sources, and tests in wheels.
- Added an offline arithmetic dataset and tiny-adapter training demo.
- Added token-boundary, local-clause, negation-aware lexical screening.
- Bundled the original 50 neutral prompts with source hashes.
- Moved generated state under the configured data root and honored external export paths.
- Retained overwritten datasets and previous native runtimes as backups.
- Removed eager model discovery from initial UI rendering.
- Added installed-wheel CI, CPU native-build CI, and pinned Action references.
- Use the pinned primary GGUF runtime by default for every architecture.
