# Installation

## Python environment

Create and activate a Python 3.11–3.13 virtual environment. Python 3.12 is used for the pinned core dependency set.

```bash
python -m pip install -e .
studio init
studio check
```

For Python 3.12, use `python -m pip install -c constraints/core.txt -e .`. Release wheels include application resources; install a wheel normally, then run `studio init`. Wheel state defaults to the operating system user data directory.

## Native GGUF backend

Install Git, CMake 3.24+, and a C++17 compiler. Windows uses Visual Studio 2022 Build Tools with Desktop development with C++. NVIDIA builds additionally require the CUDA toolkit and `nvcc` on PATH.

```bash
studio native --jobs 2
studio native --cuda --jobs 2
```

The builder fetches llama.cpp revision `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`, builds shared CPU/CUDA backends, validates the library set, and replaces the installed runtime with rollback backup. Override GPU targets with `--architectures`; the default targets the local GPU. `--cuda-root` selects a toolkit.

`STUDIO_NATIVE_RUNTIME` selects an externally built runtime. Its binary hashes become part of the direction-bank identity.

## Persona environments

```bash
python -m pip install -e ".[persona]"
python -m pip install -e ".[persona-cuda]"
python -m pip install -e ".[train]"
python -m pip install -e ".[scrape]"
```

`persona` supports CPU and CUDA BF16. `persona-cuda` adds bitsandbytes NF4. Install a CUDA-enabled PyTorch build before the CUDA extras. Training uses NF4 and requires CUDA; the tiny demo can train on CPU.

To use a separate training environment, set `[python].executable` in `studio.toml`. Set `python` in `persona-sources.json` for adapter inference. Both environments must be able to import Studio.

## Configuration

`STUDIO_HOME` selects the writable data root. `STUDIO_CONFIG` selects a TOML file. Relative `[paths]` entries resolve under the data root and workspace. `persona-sources.json` specifies external model/project roots and local HF caches. `OLLAMA_MODELS` and `HF_HOME` override model caches.

```toml
[runtime]
port = 8899
cpu_threads = 4

[python]
# executable = "/path/to/venv/bin/python"
```

Run `studio doctor --require core --require native` for GGUF prerequisites, or add `--require train` for the training environment. `studio app` binds to localhost.
