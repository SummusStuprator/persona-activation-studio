# Changelog

## 0.3.3

- Validate source, release archives, version metadata, and commit-bound checksums.
- Run installed-wheel checks across Windows/Linux and Python 3.11–3.13.
- Add a single release CI gate and retain tested distribution artifacts.
- Document branch review, tagged uploads, and public-release procedure.
- Update citation metadata and retain versioned verification records.
- Compare resumed-directory identity across Windows short-path aliases in CI.

## 0.3.2

- Read runtime policy without resolving unrelated workspace paths.
- Check the RAM floor before prefill and between decoded tokens.
- Preserve worker-exit errors when descendant cleanup reports access denied.
- Avoid descendant lookup after the parent process has exited.
- Add memory-boundary and worker-shutdown regression tests.

## 0.3.1

- Cache token text and end-of-generation metadata within each worker lifetime.
- Send full worker metadata only after model loading.
- Apply live steering only when controls or the generation phase change.
- Use storage-aware FP16 checks in the live generation path.
- Add transport cache, worker lifecycle, control revision, and cancellation regressions.

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
