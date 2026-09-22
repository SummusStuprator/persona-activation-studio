# Local readiness verification — 22 September 2026

This is an executed Windows verification snapshot, not a promise that every model, research claim or operating system has been validated.

## Executed checks

| Check | Result and scope |
|---|---|
| Software regressions | 26 model-free tests: resume identity/content, checkpoints, job outcomes, duplicate suppression, PID reuse, shared locks, RAM admission, fallback and manual offload |
| Isolated fixtures | Profile/cache corruption and invalidation checks passed without deleting live research state |
| UI navigation | All 14 destinations rendered without an exception |
| Loaded UI | All eight model-dependent pages rendered with a real Llama 3.2 1B loaded; no experiment buttons clicked by this check |
| Native GPU | Llama 3.2 1B, LFM2.5 2.6B and a Qwen 35B-A3B checkpoint generated text and passed injection/reset checks |
| Persona runtime | Four existing 4B PEFT adapters generated text, passed BF16-aware intervention checks, and returned identical baseline logits after reset |
| Training data | All 18 configured profiles had usable, non-empty filtered train/validation/test splits |
| Optimizer recovery | Tiny offline LoRA run resumed from step 2 to step 4 with exactly the same final adapter weights as uninterrupted training |
| CUDA training | Production NF4 loader and capability-preserving trainer exercised on a tiny temporary model: 14 quantized layers, two optimizer steps, changed adapter parameters |
| Streaming experiment | Exact-model directions rebuilt for the current native identity; baseline, matched-random and steered conditions produced 256 token events and passed numerical injection checks |
| Setup | Core, native libraries/CUDA, configured training environment and scraper package available; existing scraper database present |

The 35B checkpoint required an explicit 15-GPU-layer split and a 512-token context on the test machine. This is not a universal setting: its memory guard correctly deferred earlier attempts under heavier host load. Auto is conservative; manual selection now honors the requested split without bypassing host-memory admission.

## Important boundaries

Persona smoke generations are not a held-out persona-fidelity benchmark. A tiny optimizer/NF4 test is not a full-size model-training quality result. Numeric injection accuracy is not proof of subjective experience or robust semantic specificity. The separate legacy physical-preset completion retained its unverified specificity status.

The scraper session was not refreshed or used to collect new posts during this verification. A present account database does not establish that its authentication remains valid. No model weights or source datasets were downloaded for these checks.

Windows was executed. The repository includes Windows/Linux CI configuration, but hosted CI and Linux/macOS native execution were not run as part of this snapshot. Read an actual CI result before treating a workflow definition as a pass.
