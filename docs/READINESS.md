# Verification records

## 0.3.4 — 24 September 2026

Windows/Python 3.12: 80 regression tests and all 14 model-free UI routes passed from both source and a regular wheel installation outside the checkout. Checks include local-export dataset construction, input validation, archive/history regressions and documentation links. Dependency consistency, the offline dataset demo and source sealing also passed. Artifact hashes and source identity are recorded in `release.json`.

The release audit checks tracked files, distribution contents and locally reachable Git history for known private-file, credential and workstation-path patterns. A passing pattern scan does not prove absence of every possible private datum.

The [0.3.3 hosted workflow](https://github.com/SummusStuprator/persona-activation-studio/actions/runs/35859676895) passed all six Windows/Linux Python jobs, Linux native build and Release gate for commit `c57cd2d64f12516370ab12107eea01f0194ef21e`. The 0.3.4 release requires its own successful hosted matrix before publication.

Real X collection, full persona training and CUDA/model inference were not rerun for 0.3.4. Existing hardware/model records below are historical. Numerical training/steering algorithms are unchanged; model-store discovery, input validation, documentation/UI wording, package metadata/resources and release auditing changed. Software verification does not establish a new pain-related scientific result.

## Release CI

The release commit is recorded in `release.json`. Its GitHub Actions run contains the installed-wheel results for Windows/Linux and Python 3.11–3.13, the Linux CPU-native build, and the aggregate Release gate.

## 0.3.2 — 23 September 2026

Windows source: 59 regressions and 14 UI routes passed. Ubuntu 22.04 / Python 3.13: the installed wheel passed 59 regressions and 14 UI routes. The native CUDA generation/intervention/reset test and the actual CUDA NF4 persona sidebar load passed on the Windows test machine.

The 0.3.3 changes affect packaging, CI, release checks, and documentation. The numerical backend files are unchanged from 0.3.2.

## 0.3.0

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
