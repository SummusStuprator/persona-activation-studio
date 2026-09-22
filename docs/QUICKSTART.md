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

Each new run gets a unique staging directory and `studio-run.json`. The dataset path and model are pinned; profile configuration and optional anchors are copied into that run. Keep dataset revisions unchanged while a run is in progress.

Resume the single unfinished matching run with `studio train example --resume`. When multiple matching runs exist, select one with `studio train example --resume --run-dir PATH`. A complete checkpoint is required. No checkpoint means an error, not a silent fresh start. Completed runs are not resumed. Older runs without a Studio manifest are not guessed or migrated automatically.

The checkpoint-selection tests do not constitute a new end-to-end GPU training run. Existing adapters are not overwritten. Installation still requires the trainer's install gate.

## Chat and research

Open Chat, select the model source, load one model, and start without steering. Prepare model-specific directions only when needed. Activations, Emotion library, Concept builder, Experiments, J-space and Paper reproduction are separate working routes, not empty placeholders. J-space requires GGUF; the supported persona pages use the PEFT lane.

Use baseline and matched-random comparisons before interpreting a steering effect. The five-direction suite includes Pain S2 plus authored somatic-pain, burning, fire and despair extensions. Built directions, non-collapsing output and behavioral specificity are separate milestones. A 2-of-3 lexical screen is not proof of conscious experience or broad reliability.

## Resource behavior

The cooperative defaults reserve at least 10% of host RAM before loading, and yield at token/example boundaries if free RAM falls below that floor. GPU fallback in Studio is re-budgeted for its actual CPU/GPU split. These are admission estimates and boundary checks, not an absolute cap on memory used by Windows and other applications.

Do not load a model in both the legacy lab and Studio at once. The copies have separate worker locks and data roots. If another model owns your available memory, unload it deliberately in its own UI; the checks do not terminate it for you. Auto mode may fall back to CPU when CUDA is unavailable. Inspect the load plan rather than assuming Auto means GPU.

## What the checks establish

The unloaded UI test visits every navigation item and requires visible content. It does not claim that every button, training path, model architecture, or full research protocol was executed. `tests/ui_loaded_smoke.py` is a separate opt-in check requiring a real local Llama 3.2 1B and sufficient free RAM. Never turn a resource deferral into a passing model test.

The profile and pair-cache smoke tests redirect all writable fixtures into temporary directories. They must not delete the user's calibration or cache trees. Historical test reports that merely counted menu options should not be read as complete loaded-page coverage.

Source edits in a running Streamlit process with file watching disabled require a normal restart to become fully visible. Save the current experiment and unload its model first. Do not interrupt a live generation just to refresh a badge.
