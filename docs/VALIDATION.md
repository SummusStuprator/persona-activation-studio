# Validation matrix

This file records what was actually exercised before the local v0.1.0 tag.

## Clean-checkout installation

A new clone with no copied virtual environment, workspace, direction banks, or native binaries was tested on Windows.

- `scripts/setup.ps1 -Profile core` created a fresh virtual environment and installed Studio.
- `studio doctor` succeeded.
- `scripts/build-native.ps1 -Jobs 2` fetched only the pinned llama.cpp commit and compiled the CPU runtime from source.
- `tests/native_smoke.py` loaded the existing local `llama3.2:1b` GGUF through the newly built runtime and returned finite activations/logits.
- `tests/ui_smoke.py` executed all 14 central navigation pages without a Streamlit exception.
- A clean-clone Streamlit server answered `/_stcore/health` with `ok`.
- `studio research-data` downloaded the official filtered GoEmotions splits and verified pinned SHA-256 hashes.

The clean clone did not inherit the development repository's `.venv`, `native/runtime`, vectors, runs, workspace, or local configuration.

## Persona runtime

The source-format PEFT runtime was tested against an externally stored `cutyourtrachea_qwen3_1.7b_lora_v2` adapter without copying model weights into Studio.

- 28 decoder blocks, hidden width 2048.
- Finite logits and complete activation hooks.
- Model identity is bound to base, adapter, tokenizer, and runtime hashes.

The full training stack is a separate optional dependency set; the core install does not require Torch.

## Hell lab

`tests/hell_smoke.py` used a real Llama 3.2 1B GGUF and built the exact-checkpoint pain/fire/despair suite.

It then ran the neutral activation-dominant condition with an unsteered same-seed comparator.

- 96 token events were observed by the streaming callback: 48 comparator + 48 steered.
- Every steered event contained pain, fire, and despair projections.
- Final streamed projections in that run were approximately pain 2.895, fire 2.726, despair 5.373.
- Maximum observed native injection error was below 1e-3; the final token's error was about 5.6e-8.
- The callback exposed partial reasoning/final text, token index, phase, entropy, projections, active controls, and injection error during generation.

The original 8899 build was also exercised on persona adapters including `cutyourtrachea_qwen3_1.7b_lora_v2` and `5XLGroyper_qwen3_4b_strong_v2`. Their Hell directions were trained for their exact PEFT identities rather than copied from another checkpoint.

## Source/publication audit

Before the initial commit:

- `git diff --check` passed.
- tracked files were scanned for the developer machine's user path and account identifiers; none were found.
- X session databases, raw exports, model weights, adapters, generated direction banks, run records, local configs, native binaries, and virtual environments are ignored.
- no Git remote was configured.

## Cross-platform status

Windows setup, native compilation, clean-clone inference, and UI health were executed end to end.

The Unix setup/build/start scripts passed `bash -n` using Git Bash. The native bridge has Windows and ELF/macOS export paths, and IPC selects AF_PIPE on Windows and AF_UNIX elsewhere. Linux and macOS compilation were not executed on this Windows host.
