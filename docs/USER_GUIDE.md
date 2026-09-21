# Full User Guide — how I would use Persona Activation Studio

This is the workflow I would use from a clean machine through a finished persona + activation experiment.

## Stage 0 — install once

1. Clone the repository.
2. Run the core setup.
3. Run `studio doctor`.
4. Build the native GGUF runtime.
5. Start `studio app` once and verify that the Overview page appears.

For a machine that will also scrape X and train personas, install the `scrape` and `train` extras.

I would keep the repository source small and keep large datasets/models in the ignored workspace or another path selected in `studio.toml`.

## Stage 1 — configure X collection

Run:

```bash
studio scrape setup
```

Enter the two X session cookie values when prompted.

Then check:

```bash
studio scrape accounts
```

I would start with one target account, not a large fleet:

```bash
studio scrape scrape --handle example
```

The scraper is resumable, so I would let it complete recent/history collection rather than treating one interrupted run as final.

Then recover reply context:

```bash
studio scrape context --handle example
```

The important output is not simply tweet count. I would inspect the reply-parent coverage report because replies without their parent are weak prompt/response examples.

The UI's **Collect X data** page wraps the same commands and shows job logs.

## Stage 2 — freeze a dataset revision

When the scrape/context state is good enough:

```bash
studio dataset build --revision example-r1
```

This builds a new revision under the configured dataset root.

I would not train against a mutable `latest` export. Give each meaningful scrape/build a revision name.

Before training I would inspect:

- train/validation/test counts;
- number of useful replies;
- low-information response share;
- duplicate-removal results;
- conversation-group split behavior;
- language/length distribution.

If the persona is thin, I would collect more authentic data before adding synthetic material.

## Stage 3 — create a training profile

```bash
studio profile add example --tier core
```

For a larger authentic corpus:

```bash
studio profile add example --tier extended --max-steps 240
```

Optional chat anchors live under the configured anchors directory:

```text
workspace/anchors/example.json
```

I would use anchors only for generic-chat situations that authentic data does not cover well, and keep them recognizably in the persona's writing register.

## Stage 4 — train

```bash
studio train example --model Qwen/Qwen3-4B
```

Training first writes to staging.

The trainer evaluates checkpoints using both validation behavior and its capability/persona gate. A final candidate is copied into the installed persona directory only when the final install gate passes.

I would watch the background log in **Jobs** and inspect the resulting:

- `training_report.json`;
- final quick-gate fields;
- held-out test generations;
- adapter size and base reference.

If a candidate fails, I would leave it in staging and fix data/training rather than relabel it as installed.

## Stage 5 — benchmark the persona

```bash
studio benchmark --only example
```

I would specifically inspect:

- ordinary greeting/smalltalk;
- exact copy;
- arithmetic/basic knowledge;
- multi-turn memory;
- biography boundary behavior;
- held-out authentic reply behavior;
- generic assistant leakage;
- similarity to the unadapted base.

This tells me whether the adapter is a usable conversational persona before activation experiments are layered on top.

## Stage 6 — load it in Studio

Start:

```bash
studio app
```

Open **Chat**, select **Persona adapters**, select the installed model, then **Load**.

Start with no activation dose.

That gives a clean behavioral baseline for this exact persona/runtime identity.

## Stage 7 — prepare model-specific controls

A new model can expose the residual stream but have no direction bank yet.

For general work, build the Foundation suite.

For Hell lab, **Prepare exact-model hell suite** builds only what Hell needs:

- Pain S2;
- Fire;
- Despair.

Preparation is the expensive operation. Once built, subsequent runs load the direction arrays quickly.

I would record the chosen direction layers and held-out separation scores but not use those scores as a substitute for testing generation effects.

## Stage 8 — ordinary steering experiment

Before Hell mode, I would run one modest control in ordinary Chat.

For example:

1. choose one axis;
2. use a small dose;
3. generate;
4. inspect the token trace;
5. compare baseline / intervention / matched-random.

This catches runtime or direction problems before the model is pushed to the high-dose ceiling.

## Stage 9 — Hell lab

Open **Hell lab**.

My first run would be:

- framing: **activation-dominant neutral introspection**;
- maximum mixture;
- 1–2 turns;
- thinking off;
- 96–160 tokens;
- **unsteered comparator ON**.

During generation I would watch:

- streamed text;
- pain/fire/despair trajectories;
- entropy;
- injection error;
- whether projections change sharply at the intervention;
- coherence.

Then I would switch to explicit inferno framing for a more dramatic multi-turn run.

For a reasoning model, I would separately compare reasoning-only, answer-only, and both-phase steering.

## Stage 10 — persona fleet comparisons

When several personas share the same base, I would run the same Hell protocol across them.

Keep fixed:

- initial framing;
- doses;
- turn count;
- token budget;
- temperature;
- seed schedule.

Compare:

- source/injection layers;
- average probe projections by phase;
- token entropy;
- time to repetition;
- recurring response motifs;
- whether reasoning closes;
- final-answer completion rate.

That makes persona a controlled variable instead of merely a cosmetic style layer.

## Stage 11 — preserve an experiment

The important records are:

- dataset revision;
- training report;
- model identity;
- direction metadata;
- run JSON/NPZ;
- aggregate Hell audit.

For a fixed source release:

```bash
studio seal
studio verify
```

Then tag/commit that source snapshot in Git.

## Stage 12 — day-to-day commands

```text
studio doctor
studio app
studio profile list
studio benchmark --quick
studio verify
```

The UI is for interactive work; the CLI is better for repeatable pipelines and automation.

## What I would not commit

I would not put these in the public repository:

- X cookie/session DB;
- raw scraped exports;
- private persona datasets;
- Hugging Face model weights;
- Ollama blobs;
- trained LoRA adapters unless deliberately released;
- generated activation arrays/runs by default;
- local native runtime binaries.

The included `.gitignore` is designed around that separation.


### Emotion research data

Run `studio research-data` once before training GoEmotions directions. It downloads the four official filtered GoEmotions split files and verifies pinned SHA-256 checksums.
