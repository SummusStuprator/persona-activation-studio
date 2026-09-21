# Third-party components and provenance

This repository combines locally developed source from two predecessor projects and interfaces with several external projects.

## Persona pipeline

`persona/x_scraper.py`, `persona/dataset_builder.py`, `persona/trainer_v4.py`, and `persona/benchmark.py` are consolidated from the local persona/xscra development tree.

The repository-level MIT license applies to the locally developed Studio/persona-pipeline source. Third-party components and paper-derived material retain their own licenses.

## Pain Axis research material

The `paper/` directory contains the locally used reference material/datasets for:

**The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It** — Tagliabue, Dung, Berg.

The copied `paper/LICENSE` applies to that material. Keep its attribution/license with redistributed paper-derived files.

## llama.cpp

The native build scripts fetch llama.cpp from its upstream Git repository at the revision pinned in the scripts. llama.cpp source is not committed here.

Review and preserve upstream licensing when distributing native binaries.

## twscrape

The X collection extra depends on `twscrape`. It is installed from its normal package source and is not vendored.

## PyTorch / Transformers / PEFT / bitsandbytes

The persona training extra installs these as external dependencies. Model licenses remain separate from this repository: check the license of every chosen base model and every adapter/data release before redistributing weights.

## Ollama

Studio can discover GGUF files stored by a local Ollama installation. Ollama model weights are referenced in place and are not included in Git.

## Private/local artifacts

The project intentionally ignores:

- X session databases/cookies;
- raw X exports;
- local datasets;
- model weights and adapters;
- generated activation vectors and run records;
- local native runtime binaries.

Publishing the source repository therefore does not automatically publish those artifacts.
