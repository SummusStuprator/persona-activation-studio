# Installation

Persona Activation Studio separates the lightweight UI/GGUF stack from optional scraping and training dependencies.

## Requirements

Core:

- Python 3.11–3.13
- Git
- CMake 3.24+
- a C/C++ compiler supported by CMake
- Ollama if you want automatic discovery of an existing local GGUF library

For NVIDIA GGUF offload, install a working CUDA toolkit/compiler setup before the native build.

For persona training, a GPU with enough VRAM is strongly preferred. The trainer can be installed independently from the core UI.

## 1. Clone and create the environment

Windows:

```powershell
git clone <repo-url>
cd persona-activation-studio
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Profile core
```

Linux/macOS:

```bash
git clone <repo-url>
cd persona-activation-studio
bash scripts/setup.sh core
```

This creates `.venv`, installs the project in editable mode, initializes `workspace/`, creates a local `studio.toml`, and creates local persona-source configuration. None of those local state files are intended for Git.

Check the environment:

```bash
studio doctor
```

## 2. Build the native GGUF instrumentation runtime

Studio's white-box GGUF backend is a small bridge built against a pinned llama.cpp revision.

Windows CPU:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-native.ps1
```

Windows NVIDIA:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-native.ps1 -Cuda
```

Linux CPU:

```bash
bash scripts/build-native.sh
```

Linux NVIDIA:

```bash
CUDA=ON bash scripts/build-native.sh
```

The scripts clone llama.cpp under the ignored `.vendor/` directory, check out the bridge's pinned revision, build `activation_bridge` and its shared-library dependencies, and copy runtime libraries into ignored `native/runtime/`.

You can instead supply a compatible runtime directory:

```bash
export STUDIO_NATIVE_RUNTIME=/path/to/runtime
```

## 3. Optional X collection support

```bash
pip install -e ".[scrape]"
```

The scraper uses twscrape and stores its session database under the local workspace. Initialize a session interactively:

```bash
studio scrape setup
```

It asks for `auth_token` and `ct0` without echoing them. The Windows-only convenience option is:

```powershell
studio scrape setup --from-clipboard
```

after explicitly copying a Cookie request header from the browser developer tools.

## 4. Optional persona adapter runtime

If you want to load and steer PEFT/LoRA personas but do not need to train them:

```bash
pip install -e ".[persona]"
```

This installs Torch, Transformers, PEFT, Accelerate and safetensors without bitsandbytes.

## 5. Optional persona training support

```bash
pip install -e ".[train]"
```

Install a hardware-appropriate PyTorch build first if you want a particular CUDA or ROCm runtime.

The trainer uses Transformers + PEFT + bitsandbytes and expects a Hugging Face causal-LM base. It produces LoRA adapters; it does not merge the base into each persona.

## 6. Start the UI

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

Unix:

```bash
bash scripts/start.sh
```

Or:

```bash
studio app --port 8899
```

The server binds to localhost.

## Model discovery

### GGUF

By default Studio looks in:

- `OLLAMA_MODELS`, if set;
- otherwise `~/.ollama/models`.

It reads Ollama manifests and uses their model blobs in place.

### Persona adapters

Local persona discovery defaults to:

```text
workspace/models/persona
```

and resolves declared Hugging Face bases from `HF_HOME` or the normal HF cache.

For an existing external adapter collection, edit the ignored `persona-sources.json`:

```json
{
  "project_roots": ["/path/to/persona/project"],
  "model_roots": ["/path/to/persona/models"],
  "cache_roots": ["/path/to/huggingface/hub"],
  "python": "/path/to/python",
  "base_overrides": {},
  "settle_seconds": 5,
  "max_depth": 5
}
```

## Configuration

`studio.toml` controls the local working paths.

A simple portable configuration:

```toml
[paths]
workspace = "workspace"
project = "persona-project"
exports = "x_exports"
account_db = "x_accounts.db"
datasets = "datasets"
models = "models/persona"
staging = "models/staging"
anchors = "anchors"

[runtime]
port = 8899
cpu_threads = 4
```

Relative paths are resolved from the workspace/repository rather than from a particular user's home directory.

## Source sealing

For normal development, leave the checkout unsealed.

For a fixed experimental snapshot:

```bash
studio seal
studio verify
```

If a sealed source file changes afterward, worker startup reports it instead of silently treating the run as the previous release.
