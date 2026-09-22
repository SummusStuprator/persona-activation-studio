# Workflow

## Launch

Run `studio app`. Windows source installs can use `Start-Studio.cmd`. The port comes from `studio.toml`; `studio app --port 8900` overrides it.

## Verify the pipeline without an account

`studio demo` writes a deterministic arithmetic dataset with separate train, validation, and test splits. `studio demo --train --device CPU` additionally creates a tiny Qwen2 base and trains a LoRA adapter for two steps. All files remain in the workspace. The generated adapter is marked as a fixture and has no installation-gate result. Enable unrecorded adapters in the model selector to inspect it.

## Collect and train

```bash
studio scrape setup
studio scrape scrape --handle example
studio scrape context --handle example
studio dataset build --revision example-r1
studio profile add example
studio plan --dataset workspace/datasets/example-r1
studio train example --dataset workspace/datasets/example-r1
```

Inspect split counts before training. The builder uses the configured exports directory. `--overwrite` retains the previous revision in a sibling backup directory.

Training writes staging checkpoints and a content-hashed run manifest. `studio train example --resume` continues one matching unfinished run. Add `--run-dir PATH` to select a specific run. Changed data, changed configuration, and incomplete checkpoints are rejected.

## Benchmark

```bash
studio benchmark --model-root PATH --only MODEL_NAME --output-root RESULTS
```

Temperature zero uses greedy decoding; a positive temperature enables sampling. Cache identity includes model contents, prompts, sampler, held-out examples, code, and library versions. Use a new output directory when that identity changes. Results include raw generations, held-out likelihoods, CSV/JSON summaries, and an HTML report.

## Load and inspect

Choose a source in Chat, select a model, then Load. Persona Auto selects CUDA BF16, CUDA NF4, or CPU. Explicit CUDA never switches to CPU. The load plan records the device and precision. Unload releases the worker and its memory.

Prepare directions for the loaded runtime before enabling controls. CPU, CUDA BF16, and CUDA NF4 maintain separate direction banks. Use the baseline/random/target comparison to inspect generation changes.

## Update

Stop active jobs, unload the model, install the new version, and run `studio check`. For a sealed source tree, review the diff and run `studio seal` followed by `studio verify`. Restart the app after source updates.

The Jobs page can stop a selected running job. Cancellation verifies the recorded process creation time before terminating that job and its child processes.
