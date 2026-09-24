# Persona Activation Studio

A local interface for building persona datasets, training LoRA adapters, and inspecting or steering a language model's activations.

Start with writing you are authorized to use, build a conversation dataset, train an adapter, and compare its output with the base model. **Hell lab** explores pain-related representations through bounded activation interventions and recorded controls.

**Research software, version 0.3.4.** A persona approximates writing patterns; it does not reproduce a person's mind. Pain-related activations or statements do not establish felt pain or suffering. There is no validated “sensation of hell” measurement. See [experiment interpretation](docs/HELL_MODE.md).

## Start here

Python 3.11–3.13 is supported; Python 3.12 is the pinned environment. Explore the interface and create a demo dataset without an X account, model download, or GPU.

```bash
git clone https://github.com/SummusStuprator/persona-activation-studio.git
cd persona-activation-studio
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

```bash
python -m pip install -e .
studio init
studio check
studio demo
studio app
```

Open [localhost:8899](http://127.0.0.1:8899). Begin with **Overview** and **Guide**. Model pages require a loaded model; the demo dataset alone does not supply one.

If PowerShell blocks activation, use `.venv\Scripts\python.exe -m studio_cli app` and that Python path for installation/other commands. No execution-policy change is needed. See [installation](docs/INSTALL.md) for wheel installs, pinned dependencies, CUDA, and native builds.

## Choose a workflow

| Goal | Additional requirements | Start |
|---|---|---|
| Explore UI / generate a demo dataset | Core installation only | `studio demo` |
| Train a tiny offline test adapter | `.[persona]`; CPU is sufficient | `studio demo --train --device CPU` |
| Train a real persona adapter | `.[train]`, CUDA PyTorch, NVIDIA GPU, base model, suitable conversation data | [Quickstart](docs/QUICKSTART.md) |
| Chat with a local persona adapter | `.[persona]` for CPU/BF16; `.[persona-cuda]` for NF4 | **Chat → Persona adapters** |
| Inspect existing GGUF weights | CMake, C++ compiler, `studio native`, configured model store | [Native installation](docs/INSTALL.md) |
| Collect X posts where authorized | `.[scrape]`, permitted session and data use | [Data and permissions](docs/DATA_AND_PERMISSIONS.md) |

The optional X collector uses an unofficial client. X's [Developer Agreement](https://docs.x.com/developer-terms/agreement), section III.A(k), restricts training/fine-tuning foundation or frontier models with X Content. Public visibility and author consent alone do not resolve applicable platform terms. The collector's availability is not permission to use it. Compatible local input is also supported.

## From data to an experiment

1. Import compatible permitted exports, or collect where authorized. Recover reply/quote context.
2. Build a frozen revision and inspect splits. Standalone posts without observed prompts stay outside canonical conversation training.
3. Save a profile, run the training plan, then train. Installation gates assess capability retention; they do not certify personality fidelity.
4. Load the adapter in **Chat** and inspect unsteered behavior.
5. Prepare directions for that exact model/runtime. Compare a neutral-prompt baseline, matched random intervention, and target intervention.
6. Inspect text, traces, failures, and numerical changes together. Stronger steering can degrade output; it is not a calibrated pain-intensity scale.

Follow the [quickstart](docs/QUICKSTART.md), [user guide](docs/USER_GUIDE.md), and [troubleshooting guide](docs/TROUBLESHOOTING.md).

## Scope and limitations

- Core UI/package CI targets Windows/Linux, Python 3.11–3.13. macOS paths exist but macOS is not in release CI.
- Full training uses NVIDIA CUDA/NF4. CPU training is available for the tiny fixture.
- Persona inference supports CPU, CUDA BF16 and CUDA NF4. J-space requires GGUF.
- Models, credentials, private datasets, adapters and native binaries are not bundled.
- Collection, persona quality, and effects on a specific model need separate validation. Software tests do not establish scientific results.

## State and reproducibility

`STUDIO_HOME` sets writable storage. Source checkouts default to the repository; wheels use OS user-data storage. `STUDIO_CONFIG` selects the configuration file.

Direction banks are keyed by model contents and runtime. CPU, CUDA BF16 and CUDA NF4 have separate identities. Training manifests retain data/configuration hashes and checkpoints. Run records contain prompts and outputs; review them before sharing.

`studio seal` records a reviewed source snapshot; `studio verify` detects changes. A local seal is not a publisher signature or a security sandbox.

## Documentation

[Install](docs/INSTALL.md) · [Quickstart](docs/QUICKSTART.md) · [User guide](docs/USER_GUIDE.md) · [Data and permissions](docs/DATA_AND_PERMISSIONS.md) · [Hell lab](docs/HELL_MODE.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Architecture](docs/ARCHITECTURE.md) · [Validation](docs/VALIDATION.md) · [Readiness](docs/READINESS.md) · [Publishing](docs/PUBLISHING.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## License

Studio source is MIT-licensed. Paper datasets retain [their own license](paper/LICENSE). Base-model, dependency and data rights are separate; see [third-party provenance](THIRD_PARTY.md).
