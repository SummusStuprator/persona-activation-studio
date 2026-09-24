# From writing to a persona experiment

## 1. Verify installation

Follow [INSTALL.md](INSTALL.md), then run:

```bash
studio init
studio check
studio demo
studio app
```

The demo writes a deterministic arithmetic dataset with separate training, validation and test examples. It tests the pipeline, not persona quality.

To test an adapter without downloading weights, install `.[persona]` and run `studio demo --train --device CPU`. This constructs a tiny Qwen2 model and trains for two optimizer steps. Its adapter has no installation-gate result; enable unrecorded adapters in the selector to inspect it. Expect low-quality text.

## 2. Prepare permitted input

Read [DATA_AND_PERMISSIONS.md](DATA_AND_PERMISSIONS.md). Compatible local exports can be used without an account connection.

If collection and downstream use are permitted, install `.[scrape]` and use an authorized session:

```bash
studio scrape setup
studio scrape accounts
studio scrape scrape --handle example --recent-only
studio scrape context --handle example
studio scrape status --handle example
```

Use a handle, not a URL. Setup is interactive in a terminal; do not paste secrets into issues or chat. A local account entry does not prove X will accept future requests. Collection may be incomplete because of missing posts, expired sessions, limits or platform changes.

Omit `--recent-only` for historical search, or specify `--start YYYY-MM-DD --end YYYY-MM-DD`. Respect access restrictions. `studio scrape export --handle example` rebuilds CSV exports from saved records.

## 3. Build and inspect a revision

```bash
studio dataset build --revision example-r1
```

With default source-checkout configuration, the result is `workspace/datasets/example-r1`. With a custom data root, use the path printed by the command. Inspect:

- `summary_by_profile.csv`: posts, usable conversations and split counts;
- `manifest.json`: source, grouping parameters, validation and counts;
- `profiles/example/sft/portable/core/{train,validation,test}.jsonl`: recommended text-only conversations;
- `profiles/example/queues/`: unresolved context, media-dependent and low-information material.

Related conversations and duplicates remain in one split. Many posts in few conversations can leave validation/test empty. Fix input before training; do not copy held-out examples into training. Standalone posts do not automatically become supervised examples.

Use a new revision name to preserve prior input. `--overwrite` moves the previous revision to a sibling backup, which still contains the original private data.

## 4. Plan and train

Install `.[train]` with CUDA-enabled PyTorch. Full training requires NVIDIA CUDA/NF4. The default base is `Qwen/Qwen3-4B`; weights must be local or downloaded on first use, subject to their license/access requirements.

```bash
studio profile add example
studio plan --dataset workspace/datasets/example-r1
studio train example --dataset workspace/datasets/example-r1
```

The profile name must match the dataset profile. Use `studio profile add my-adapter --dataset-profile example` when names differ. In the UI, click **Save profile config** before **Start training**.

The plan checks every configured profile and exits unsuccessfully if any is not ready. It reports split sizes and step budgets; it does not load a model or establish that your GPU has sufficient VRAM.

Runs live under `workspace/models/staging/`. Their manifests pin data, configuration, anchors and model reference. Passing the installation gate copies the adapter/reports into `workspace/models/persona/`; failing leaves the candidate in staging. Inspect `training_report.json`: successful process exit alone does not mean an adapter was installed.

```bash
studio train example --resume
```

Resume requires one compatible unfinished run and a valid checkpoint. Use `--run-dir PATH` if several match. Changed data/configuration or incomplete checkpoints are rejected.

## 5. Evaluate and load

```bash
studio benchmark --only example
```

The benchmark compares installed adapters and their bases, writing generations, held-out likelihoods, summaries and HTML under `workspace/benchmark/`. `--only` accepts profile or model-directory names. Supply `--base-model` when the default Qwen3-4B fallback is inappropriate.

Use a new `--output-root` when model, data, sampler, code or library identity changes. Temperature zero is greedy decoding. Held-out loss and capability checks do not certify human identity or universal persona fidelity.

In **Chat → Persona adapters**, select and **Load** a model. Auto chooses CUDA BF16, CUDA NF4 or CPU based on availability/memory. Explicit CUDA fails instead of switching to CPU. The sidebar reports the actual runtime.

## 6. Inspect and steer

First generate an unsteered response. Prepare exact-runtime directions, then open **Hell lab**. Start with the neutral prompt, independent trials and automatic dose. Enable both comparison checkboxes and inspect the prompt/control plan.

The loop's baseline/random comparators cover the first turn only. Use specificity validation for condition-matched multi-seed screening. Recurrent runs feed earlier output back; escalating presets also change dose. Read [HELL_MODE.md](HELL_MODE.md) before interpreting scores.

## 7. Stop and update

Chat can stop active live generation. Jobs can stop a selected subprocess; it is not a universal cancel button for synchronous lab operations. Lab runs have turn/token/repetition bounds. Let them finish before unloading/updating.

Stop jobs, unload the model, install the update and run `studio check`. Review changes before `studio seal` and `studio verify`. Restart the app.
