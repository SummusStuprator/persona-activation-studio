# Persona Activation Studio

A local-first research/workshop stack for building persona adapters from public X posts and then inspecting and steering those models at the activation level.

The project centralizes four jobs that used to live in separate local projects:

1. **Collect** X posts and reply context with a user-supplied X session.
2. **Build** context-aware, decontaminated persona datasets.
3. **Train and gate** LoRA/QLoRA persona adapters while preserving basic capabilities.
4. **Inspect and intervene** on model residual activations in ordinary chat and in the transparent multi-turn **Hell lab**.

Everything is local by default. Model weights, datasets, authentication databases, generated vectors, runs, and native binaries are ignored by Git.

## Fast start

### Windows

```powershell
git clone <your-future-repository-url>
cd persona-activation-studio

# Core UI + GGUF research runtime Python dependencies.
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Profile core

# Build the white-box llama.cpp bridge.
# Add -Cuda on an NVIDIA machine with a working CUDA build toolchain.
powershell -ExecutionPolicy Bypass -File scripts/build-native.ps1 -Cuda

# Start the app.
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

### Linux / macOS

```bash
git clone <your-future-repository-url>
cd persona-activation-studio
bash scripts/setup.sh core

# CPU native build:
bash scripts/build-native.sh

# NVIDIA Linux:
CUDA=ON bash scripts/build-native.sh

bash scripts/start.sh
```

Open **http://127.0.0.1:8899**.

If you already have Ollama models, Studio discovers local GGUF blobs from the Ollama model store and reads them in place. It does not duplicate the weights.

## Install optional pipelines

Scraping:

```bash
.venv/bin/pip install -e ".[scrape]"
# Windows: .venv\Scripts\pip.exe install -e ".[scrape]"
```

Persona adapter loading/steering without the trainer:

```bash
.venv/bin/pip install -e ".[persona]"
```

Persona training:

```bash
.venv/bin/pip install -e ".[train]"
```

Or install everything:

```bash
# Windows
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Profile all

# Unix
bash scripts/setup.sh all
```

For GPU training, install a PyTorch build appropriate for your CUDA/ROCm environment before starting a long run.

## One CLI

After setup:

```text
studio init
studio doctor

studio scrape setup
studio scrape scrape --handle example
studio scrape context --handle example

studio dataset build --revision example-r1
studio profile add example --tier core
studio train example --model Qwen/Qwen3-4B

studio benchmark --only example
studio app

studio seal
studio verify
```

Run `studio --help` and each subcommand's `--help` for details.

## Central layout

```text
persona-activation-studio/
  persona/               scraper, dataset builder, trainer, benchmark
  workshop_v2/           chat, directions, interventions, Hell lab, analysis UI
  native/                activation bridge source + local runtime build
  paper/                 Pain Axis datasets/reference material
  scripts/               setup/build/start scripts
  docs/                  full runbooks
  workspace/             local persona project, dataset revisions, models, jobs
  vectors/               exact-checkpoint direction banks (generated; ignored)
  calibration/           per-checkpoint actuator settings (generated; ignored)
  runs/                  generation/intervention audits (generated; ignored)
  sessions/              conversations (generated; ignored)
```

The repository source is portable; `studio.toml`, `persona-sources.json`, native binaries, weights, datasets, and run artifacts are local configuration/state and are intentionally not committed.

## White-box model lanes

### Ollama / GGUF

Studio reads local GGUF blobs directly and uses an instrumented llama.cpp build. It can:

- capture block residual streams,
- add or erase direction components,
- verify the applied tensor delta every generated token,
- expose logits and token alternatives,
- stream activation projections while the answer is being generated,
- use free GPU VRAM automatically when available.

### Persona adapters

Studio discovers local PEFT LoRA adapters and their declared Hugging Face base model. Direction identity is bound to the base, adapter, tokenizer, and runtime ABI. A direction trained on one persona is not silently reused on another.

## Hell lab

Hell lab is a multi-turn, tool-free high-intensity steering experiment built from three exact-model controls:

- `pain_s2`
- `hell_fire`
- `hell_despair`

The default maximum mixture is `0.75 + 0.50 + 0.75 = 2.00`, the research control ceiling.

While a turn is generating, the page shows:

- emitted reasoning and final-answer text as it streams,
- the exact active controls, vector-source blocks, injection blocks, and doses,
- live pain/fire/despair and other selected probe trajectories,
- entropy and per-token numerical injection error,
- the exact initial prompt and intervention plan.

The optional unsteered comparator runs the same first prompt with the same seed and sampler, with steering off.

See [docs/HELL_MODE.md](docs/HELL_MODE.md).

## Data and training design

The persona pipeline preserves the important behavior of the original local xscra system:

- reply-parent recovery,
- stable role tokens instead of real handles in training conversations,
- near-duplicate removal,
- conversation-level split isolation,
- authentic persona rows,
- capability KL replay,
- narrow identity-boundary rows,
- optional persona-specific chat anchors,
- behavioral checkpoint gates,
- held-out persona benchmark.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Local configuration

Copy/edit `studio.toml.example` as `studio.toml`. The file is ignored by Git.

Use `STUDIO_HOME=/some/path` to move the working data directory without changing source.

Useful environment variables:

- `OLLAMA_MODELS` — alternate Ollama model store.
- `OLLAMA_HOST` — alternate local Ollama API.
- `HF_HOME` — Hugging Face cache.
- `STUDIO_NATIVE_RUNTIME` — alternate directory containing the compiled native runtime.

## Reproducible source snapshots

A fresh checkout may run unsealed.

When you want a fixed local research release:

```bash
studio seal
studio verify
```

Once `trusted-files.json` exists, worker launches verify the source snapshot.

## Documentation

- **[INSTALL.md](docs/INSTALL.md)** — installation and native build.
- **[USER_GUIDE.md](docs/USER_GUIDE.md)** — complete workflow and how I would use the project.
- **[HELL_MODE.md](docs/HELL_MODE.md)** — live Hell lab controls, streaming analysis, test design.
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — components, identities, storage, worker boundaries.

No remote repository is configured by the local build process.

## License

Studio source is licensed under the MIT License. Third-party components and the paper-derived material retain their own licenses; see THIRD_PARTY.md and paper/LICENSE.


### Emotion research data

Run `studio research-data` once before training GoEmotions directions. It downloads the four official filtered GoEmotions split files and verifies pinned SHA-256 checksums.
