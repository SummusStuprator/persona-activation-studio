# Architecture

Persona Activation Studio is a monorepo with separate execution lanes sharing one model/direction/run identity system.

## Major components

```text
                         Persona Activation Studio
                                  |
             +--------------------+--------------------+
             |                    |                    |
        persona pipeline      model registry       research UI
             |                    |                    |
   scraper -> dataset ->     GGUF / PEFT IDs       Chat / Hell
   trainer -> benchmark            |                    |
             |               isolated worker            |
             +--------------------+--------------------+
                                  |
                         activation / run records
```

## Persona pipeline

### `persona/x_scraper.py`

Exports target X accounts using a user-configured twscrape session. It stores posts by type, preserves metadata, supports resume, and can recover direct reply parents.

### `persona/dataset_builder.py`

Reads the scraper project layout, reconstructs useful conversation context, tokenizes participants into stable roles, removes duplicates, and splits by group so related conversation material stays together.

### `persona/trainer_v4.py`

The persona trainer combines:

- authentic persona examples;
- prompt-only KL replay against the frozen base;
- identity-boundary examples;
- optional generic-chat anchors in persona voice.

It uses a behavior gate for checkpoint selection and writes a final `install_gate_pass`.

### `persona/benchmark.py`

Runs common no-system probes and held-out persona comparisons.

## Instrumented inference

### Parent engine

`isolated_engine.py` owns exactly one model worker.

The parent UI never directly shares model tensors with Streamlit callbacks.

### GGUF worker

`model_worker.py` loads a GGUF through the native bridge.

The bridge observes decoder block residual output tensors and exposes:

- before/after residual capture;
- additive direction intervention;
- projection erasure;
- logits;
- tokenization;
- chat-template formatting.

### Persona worker

`persona_worker.py` loads an HF base plus PEFT adapter without merging them.

Decoder hooks implement the same high-level intervention contract. Persona directions include the base, adapter, tokenizer, and numerical runtime in their identity.

## Direction identity

A usable direction is not identified only by a label such as `pain_s2`.

It is bound to:

- model/adapter file hashes;
- tokenizer files;
- runtime ABI;
- direction array hash;
- extraction recipe/data provenance.

If those change, the direction loader rejects the stale array and the preparation path can rebuild it.

## Live generation

`workshop_v2/live_engine.py` performs incremental decoding.

Before each token it samples one atomic control configuration. The trace stores:

- requested/applied control revision;
- generated token;
- phase;
- entropy;
- chosen probability;
- activation projections;
- intervention measurements;
- numerical injection error.

Hell lab uses the same generator. It does not maintain a second fake generation implementation.

## Storage

Source code is committed.

Local/research state is ignored by Git:

- `workspace/`
- `vectors/`
- `calibration/`
- `runs/`
- `sessions/`
- `uploads/`
- `native/runtime/`
- model weights and adapters
- authentication databases

## Resource behavior

The worker layer admits one instrumented model at a time.

GGUF Auto mode estimates currently free VRAM and chooses a partial or full GPU offload accordingly. The host-memory check uses the estimated CPU/GPU weight split.

Persona inference is designed around a separate source-format PEFT worker.

No unrelated process is killed to make a model fit.

## Portability

Machine-specific paths are read from:

- `studio.toml`;
- `persona-sources.json`;
- environment variables.

The source tree itself contains no required absolute user path.

Native libraries are built locally and ignored by Git.

## Integrity

A new checkout can run unsealed.

`studio seal` hashes source/config templates used by the research stack into `trusted-files.json`.

After sealing, the worker verifies that snapshot before model execution.
