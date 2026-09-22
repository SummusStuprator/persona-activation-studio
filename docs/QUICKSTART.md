# Everyday workflow

## Start and check

From an installed checkout, run `studio check` for isolated software checks, then `studio app` to open the app. Without an activated environment, use `.venv\Scripts\python.exe -m studio_cli check` and `.venv\Scripts\python.exe -m studio_cli app` on Windows.

The start scripts use the port in `studio.toml`. Override it with `studio app --port 8900`, `scripts/start.ps1 -Port 8900`, or `STUDIO_PORT=8900 bash scripts/start.sh`. Use separate ports when the legacy 8899 lab is already running. Starting the UI does not load a model.

`studio check` does not train models, scrape accounts, or run experiments. It stops on failure and checks training-run selection, resource headroom, calibration status, isolated caches/profiles, recipe structure, and all 14 navigation destinations without loading weights.

## Train without accidentally restarting

Create a frozen dataset revision and a profile, then start a run:

```text
studio dataset build --revision example-r1
studio profile add example
studio train example --dataset workspace/datasets/example-r1
```

Each new run gets a unique staging directory and `studio-run.json`. The dataset path, content hash and model are pinned; profile configuration and optional anchors are copied into that run and content-hashed. Resume refuses edited datasets or pinned configuration. Keep dataset revisions unchanged while a run is in progress.

Resume the single unfinished matching run with `studio train example --resume`. When multiple matching runs exist, select one with `studio train example --resume --run-dir PATH`. A complete checkpoint is required. No checkpoint means an error, not a silent fresh start. Completed runs are not resumed. Older runs without a Studio manifest are not guessed or migrated automatically.

A tiny offline LoRA test also exercises the actual capability-preserving trainer, optimizer checkpoint, scheduler and RNG recovery; resumed and uninterrupted weights match. This is a mechanics test, not a full-sized GPU training quality benchmark. Existing adapters are not overwritten. Installation still requires the trainer's install gate.

## Chat and research

Open Chat, select the model source, load one model, and start without steering. Prepare model-specific directions only when needed. Activations, Emotion library, Concept builder, Experiments, J-space and Paper reproduction are separate working routes, not empty placeholders. J-space requires GGUF; the supported persona pages use the PEFT lane.

Use baseline and matched-random comparisons before interpreting a steering effect. The five-direction suite includes Pain S2 plus authored somatic-pain, burning, fire and despair extensions. Built directions, non-collapsing output and behavioral specificity are separate milestones. A 2-of-3 lexical screen is not proof of conscious experience or broad reliability.

## Resource behavior

The cooperative defaults reserve at least 10% of host RAM before loading, and yield at token/example boundaries if free RAM falls below that floor. GPU fallback in Studio is re-budgeted for its actual CPU/GPU split. These are admission estimates and boundary checks, not an absolute cap on memory used by Windows and other applications.

Do not load a model in both the legacy lab and Studio at once. The copies retain separate data roots but now share an OS-released model lock for this user, including Studio training. A second load is refused rather than running two instrumented models concurrently. If another model owns your available memory, unload it deliberately in its own UI; the checks do not terminate it for you. Auto mode may fall back to CPU when CUDA is unavailable. Inspect the load plan rather than assuming Auto means GPU.

## What the checks establish

The unloaded UI test visits every navigation item and requires visible content. It does not claim that every button, training path, model architecture, or full research protocol was executed. `tests/ui_loaded_smoke.py` is a separate opt-in check requiring a real local Llama 3.2 1B and sufficient free RAM. Never turn a resource deferral into a passing model test.

The profile and pair-cache smoke tests redirect all writable fixtures into temporary directories. They must not delete the user's calibration or cache trees. Historical test reports that merely counted menu options should not be read as complete loaded-page coverage.

Source edits in a running Streamlit process with file watching disabled require a normal restart to become fully visible. Save the current experiment and unload its model first. Do not interrupt a live generation just to refresh a badge.

## One-click launch and readiness

On Windows, double-click `Start-Studio.cmd`. It reuses an answering local UI or starts Studio, then opens the configured address. It does not start model inference automatically.

`studio doctor --require core --require native --require train --require scrape` verifies the required environments and native library loading. Omit optional requirements on machines without those pipelines. Scraper database presence does not certify current X authentication. The supported training recipe currently requires a CUDA-capable training environment.

`studio plan` validates all configured persona splits without training. The Train persona page lets you choose an existing profile, select a dataset revision explicitly, or resume an unfinished run. Configure a separate training interpreter with `[python] executable` in the ignored `studio.toml`; collection continues to use the core environment. Existing data can be referenced with `[paths] project` and `[runtime] default_dataset`; no weight or dataset copying is required.

The Jobs page records successful, failed and interrupted outcomes with exit codes. Identical active jobs are not launched twice. Logs are read using a bounded tail. A successful process exit is not a behavioral research verdict.

## Native builds and recovery

Use `scripts/build-native.ps1 -Cuda` or `CUDA=ON bash scripts/build-native.sh` for a supported NVIDIA toolchain. CPU and CUDA builds use separate directories. The build includes backend plugins explicitly and keeps the old installed runtime until the replacement passes a library-load check.

Windows CUDA dependencies are resolved from the installed toolkit (`CUDA_PATH` or `nvcc` on PATH, including `bin/x64`). They do not need to be duplicated beside Studio's libraries. Native deployment refuses while a model or training run holds the shared slot. Previous libraries are retained under `backups/native-runtime-*`.

Changing native binaries changes the runtime identity. Existing research records remain intact, but old direction banks may need to be rebuilt for the new identity. Never relabel old arrays to bypass that check.

## Additional opt-in checks

`tests/optimizer_resume_smoke.py` requires the training environment and uses a tiny random model offline. `tests/persona_fleet_smoke.py` and `tests/native_fleet_smoke.py` accept explicit `--model` arguments and an `--output` report path; the latter accepts `--require-gpu`. They run bounded real generations, test numeric interventions, verify reset behavior and close every worker. They do not establish persona fidelity, general behavioral specificity or subjective experience.

The repository includes a Windows/Linux software-check workflow. Adding a workflow is not evidence that hosted CI or either operating system has already passed; consult an actual run before making that claim.
