# Persona Activation Studio

Collect public writing, build conversation datasets, train LoRA adapters, and inspect or modify model activations during inference.

Studio has two model backends: instrumented llama.cpp for GGUF files and Transformers/PEFT for adapters. The interface includes chat, activation plots, direction fitting, paired comparisons, and the Hell lab experiment runner.

## Install

Python 3.11–3.13 is supported. The pinned core environment uses Python 3.12.

```bash
git clone https://github.com/SummusStuprator/persona-activation-studio.git
cd persona-activation-studio
python -m venv .venv
```

Activate `.venv` with `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS.

```bash
python -m pip install -e .
studio init
studio check
studio app
```

The default address is `http://127.0.0.1:8899`. Set `[runtime].port` in `studio.toml` to use another port. `Start-Studio.cmd` opens an installed Windows checkout.

For the pinned Python 3.12 environment, add `-c constraints/core.txt` to the install command.

A built wheel can be installed with `python -m pip install PATH_TO_WHEEL`. It includes configuration templates, documentation, tests, paper datasets, and native bridge sources.

## Models and hardware

| Backend | Devices | Weight format |
|---|---|---|
| GGUF | CPU, NVIDIA CUDA, partial GPU offload | Existing Ollama GGUF blobs |
| Persona adapters | CPU, CUDA BF16, CUDA NF4 | Local safetensors base and PEFT LoRA |
| Training | NVIDIA CUDA | NF4 base and trainable LoRA |
| Demo training | CPU or CUDA | Tiny generated Qwen2 fixture |

For GGUF inference, install CMake and a C++ toolchain, then build the runtime:

```bash
studio native                 # CPU
studio native --cuda          # NVIDIA; requires nvcc
```

Install a CUDA-enabled PyTorch build in the model environment before installing the optional CUDA pipelines.

```bash
python -m pip install -e ".[persona]"       # CPU or CUDA BF16 adapters
python -m pip install -e ".[persona-cuda]"  # Adds NF4 adapter inference
python -m pip install -e ".[train]"         # Persona training
python -m pip install -e ".[scrape]"        # X collection
```

In **Chat → Persona adapters**, choose **Auto**, **CUDA**, or **CPU**. Auto tries CUDA BF16, CUDA NF4, then CPU according to available memory. Explicit CUDA reports insufficient VRAM rather than switching to CPU. The loaded configuration is shown in the sidebar.

## Dataset and training workflow

```bash
studio scrape setup
studio scrape scrape --handle example
studio scrape context --handle example
studio dataset build --revision example-r1
studio profile add example
studio plan --dataset workspace/datasets/example-r1
studio train example --dataset workspace/datasets/example-r1
studio benchmark --only example
```

Training runs retain their dataset hash, configuration, anchors, model reference, and checkpoints. Resume with `studio train example --resume`; use `--run-dir PATH` when several unfinished runs match. Installed adapters must pass the trainer's installation gate.

To exercise the pipeline without an account or model download:

```bash
studio demo
studio demo --train --device CPU
```

## State and reproducibility

`STUDIO_HOME` sets the writable data root. Source checkouts default to the repository directory; wheel installs default to the operating system's user data directory. `STUDIO_CONFIG` selects a configuration file. Models remain in their configured caches or adapter directories.

Direction banks are keyed by model files and numerical runtime. CPU, CUDA BF16, and CUDA NF4 have separate identities. Run records contain the prompt, sampler, controls, layers, token trace, and intervention measurements.

`studio seal` records a reviewed source snapshot. `studio verify` checks it. After editing a sealed installation, rerun the tests and reseal before launching workers.

## Documentation

[Installation](docs/INSTALL.md) · [Workflow](docs/QUICKSTART.md) · [Architecture](docs/ARCHITECTURE.md) · [Experiments](docs/HELL_MODE.md) · [Release checks](docs/VALIDATION.md) · [Publishing](docs/PUBLISHING.md) · [Contributing](CONTRIBUTING.md)

## License

Studio source is MIT-licensed. Paper datasets retain the license in `paper/LICENSE`. Dependency, base-model, and dataset licenses are listed separately in [THIRD_PARTY.md](THIRD_PARTY.md).
