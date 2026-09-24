# Troubleshooting

Run `studio check` for model-free checks. Use `studio doctor --require core` and add `--require native`, `--require train` or `--require scrape` as needed. Doctor checks availability, not account authentication or scientific validity.

| Symptom | Action |
|---|---|
| `studio` not recognized | Activate the environment or use its Python with `-m studio_cli` |
| PowerShell blocks activation | Invoke `.venv\Scripts\python.exe` directly |
| Missing optional module | Install the matching extra in the environment actually running that component |
| Training fails after core install | Full training requires CUDA PyTorch, bitsandbytes, NVIDIA GPU and valid data |
| Native runtime/compiler missing | Install Git, CMake, C++ toolchain; run `studio native`. CUDA also needs toolkit/`nvcc` |
| Port occupied | Use `studio app --port 8900` |
| No models | Check intended model/cache roots or `OLLAMA_MODELS`; weights are not bundled |
| Adapter not listed | Check local base, complete adapter and installation gate. Enable unrecorded adapters for demo fixtures |
| Explicit CUDA fails | Check free VRAM/dependencies; select CPU explicitly if acceptable |
| Source seal mismatch | Stop jobs, review diff, test, then `studio seal` and `studio verify`; restart |
| Missing parent file | Provide `reply_context/parents.jsonl` or `parents.csv`; absent context reduces usable examples |
| Many posts, few examples | Inspect unresolved/standalone queues; canonical SFT uses observed conversations |
| Plan profile error | Check profile name/tier/splits. The plan checks every configured profile |
| No installed adapter after training | Inspect staging `training_report.json` and its gate |
| Resume ambiguous/rejected | Choose `--run-dir`; preserve pinned input and valid checkpoints |
| Benchmark cache rejected | Use a new `--output-root` after identity-changing inputs |
| Direction bank unavailable | Prepare again for the actual model/runtime; do not rename banks to bypass checks |
| Hell output repetitive/empty | Inspect token budget/stop reason; lower dose. Reasoning shares the answer budget. This is not greater-pain evidence |
| Collection incomplete/stopped | Inspect Jobs/status/session validity; respect restrictions and limits |

Jobs stops selected subprocesses; Chat has a separate live-generation stop control. Synchronous lab operations are bounded but do not currently have the same stop button.

Before reporting, remove secrets/private text/workstation paths. Include version, OS, exact command, redacted error and synthetic input. Never upload the entire workspace.
