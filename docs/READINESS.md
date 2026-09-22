# 0.3.0 verification

Executed on 22 September 2026.

| Component | Executed check | Result |
|---|---|---|
| Windows / Python 3.12 | Fresh wheel install outside the checkout; init, checks, demo, seal, verify, dependency consistency | Passed |
| Linux / Python 3.13 | Fresh wheel install and 45 regression tests | Passed |
| Linux native | CPU bridge compiled from packaged sources; Llama 3.2 1B generation, injection, reset | Passed |
| Windows native | Primary CUDA runtime; Llama 3.2 1B generation, injection, reset | Passed |
| Persona CPU | Qwen3 1.7B adapter; generation, injection, reset | Passed |
| Persona CUDA BF16 | Qwen3 1.7B adapter; generation, injection, reset | Passed |
| Persona CUDA NF4 | Qwen3 4B adapter; generation, injection, reset | Passed |
| Persona Auto | Selected CUDA NF4 for the 4B adapter | Passed |
| CUDA direction fitting | Fit 40 text examples, reload the bank, generate 26 tokens under a control; zero excess rounding error | Passed |
| Demo | Local dataset and two-step tiny Qwen2 LoRA training | Passed |
| Benchmark | Base and adapter generations, two held-out examples, NLL, JSON/CSV/HTML export | Passed |
| Source seal | Changed contents detected even with unchanged file size and timestamp | Passed |
| Persona UI | Actual Load persona model action with CUDA NF4 | Passed |
| Optimizer recovery | Resumed step 2 to step 4; final adapter weights identical | Passed |
| GPU trainer | Two NF4 optimizer steps across 14 quantized layers | Passed |
| Model-free UI | All 14 destinations | Passed |

The Windows CUDA environment used Torch 2.11.0+cu128, Transformers 4.57.1, PEFT 0.19.1, and bitsandbytes 0.50.0 on an RTX 5070 Laptop GPU. CPU, CUDA BF16, and CUDA NF4 records retain separate runtime identities.

The release workflow builds regular packages on Windows/Linux for Python 3.11, 3.12, and 3.13 and includes a Linux CPU-native build. Local commands and hosted workflow executions are recorded separately.
