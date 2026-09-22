# Architecture

## Components

`persona.x_scraper` collects posts and reply parents. `persona.dataset_builder` groups conversations, normalizes participants, removes duplicates, and writes isolated splits. `persona.trainer_v4` trains LoRA adapters with capability replay and checkpoint gates. `persona.benchmark` compares adapters and their base models.

`studio_app` renders the interface. `isolated_engine` owns a model worker through authenticated local IPC. GGUF workers call the native residual bridge; persona workers use decoder hooks in Transformers/PEFT. The web process does not own model tensors.

## Execution

A per-user launch lock serializes startup. A model lock is held by one worker or training job. Lock files are released by the operating system when their process exits. `STUDIO_LOCK_DIR` provides an explicit lock namespace.

GGUF Auto estimates available VRAM and retries lower offload after load failures. Persona Auto chooses BF16, NF4, then CPU. Explicit modes retain their selected device/precision. Every load checks host-memory reserve; long operations check available RAM at example/token boundaries.

Live generation records prompt, sampler, token IDs, logits-derived measurements, controls, injection layers, activation projections, and stop reason. Intervention cleanup and cache reset run on completion and failure.

## Identity

Persona identity hashes base, adapter, and tokenizer contents, then binds the source identity to the inference ABI. The ABI records device, quantization, numerical precision, relevant ML versions, and backend code. GGUF runtime identity hashes all deployed shared libraries. Stored arrays have separate SHA-256 checks.

## Files

`studio_paths.CODE_ROOT` contains importable code. `ASSET_ROOT` contains packaged templates, documentation, datasets, and native sources. `data_root()` resolves writable state from `STUDIO_HOME` or the installation default.

State includes `workspace/`, `vectors/`, `calibration/`, `paper_reproduction/`, `runs/`, `sessions/`, `cache/`, and `native/runtime/`. None is required in Git. Models are referenced from their existing configured stores.

## Recovery

Training manifests pin dataset and configuration contents. Jobs record process creation time and exit status. Native deployment stages a complete runtime before swapping directories and retains the previous runtime. A source seal rejects changed or missing application files before worker startup.
