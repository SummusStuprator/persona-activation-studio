# User guide

The normal workflow is **Collect X data → Build dataset → Train persona → Chat → Hell lab**. An offline fixture is available through `studio demo`. See the [quickstart](QUICKSTART.md) for commands.

## Navigation

| Page | Purpose | Requirements / limits |
|---|---|---|
| Overview | Workspace, model counts, software-check job | Counts refresh on request |
| Collect X data | Collect/resume, recover parents, rebuild exports | Collector extra and permitted account/data use |
| Build dataset | Freeze a named revision | Compatible exports |
| Train persona | Save profile, train/resume | Training extra, CUDA, base model and valid splits |
| Chat | Load, generate and control activations | GGUF runtime or persona inference extra |
| Hell lab | Fit directions and run bounded comparisons | Exact-runtime direction suite |
| Activations | Inspect activation/projection readouts | Loaded model and directions |
| Emotion library | Prepare and inspect emotion directions | Some recipes need `studio research-data` |
| Concept builder | Define positive/negative contrasts | Separate training/evaluation examples |
| Experiments | Compare interventions | Prepared model/directions |
| J-space | Jacobian/frame analysis | Native GGUF; substantial RAM/storage may be needed |
| Paper reproduction | Local adaptations of published protocols | Not every upstream experiment is bundled |
| Jobs | Subprocess status, logs and stop control | Current workspace's job records |
| Guide | Packaged documentation | No model needed |

## Data quality

Canonical training contains observed replies or quote commentary with recovered context. Standalone posts remain in a style corpus/queue; Studio does not invent prompts. Thousands of standalone posts can yield little conversation training data.

**Core** contains text-only examples. **Extended** includes text-bearing posts with media, without supplying missing visual meaning. Inspect examples before selecting it.

**Portable** replaces handles with role tokens; **verbatim** retains handles. Portable data still contains text, IDs, dates, URLs and potentially identifying details. It is not anonymized.

Train on training data, tune decisions against validation, and reserve test data for final evaluation. Conversation/duplicate grouping reduces leakage but does not prove semantic independence.

## Training and evaluation

Profiles live in `workspace/profiles.json` and select dataset profile, tier and optional row/length/step caps. Saving a profile does not train it.

Run manifests/checkpoints preserve recovery information. Optimizer checkpoints must be trusted. Installation gates assess capability retention and generated behavior. Passing candidates appear under configured models; failed candidates remain in staging.

Compare base/adapted held-out likelihood, task retention, style, copied passages and factual errors. A benchmark score is not a percentage of a human identity captured.

## Models and devices

GGUF uses the instrumented llama.cpp bridge built by `studio native`. Default discovery references the configured Ollama model store; core installation downloads no weights. `OLLAMA_MODELS` selects the store to inspect.

Persona inference uses local safetensors with Transformers/PEFT. `persona-sources.json` configures adapter roots and base-model caches. An adapter alone is not a complete model.

The sidebar's actual device/precision is authoritative. A CPU direction bank cannot simply be reused under CUDA NF4. Stop live generation before changing model or moving to another model workspace. Unload to release memory.

## Directions, doses and evidence

A direction is estimated from text contrasts. Projection measures alignment with that direction; it is not an emotion sensor. Held-out AUC measures labeled-data separation, not subjective experience.

The source layer supplies the direction; the injection layer receives it. Preserve both. Dose controls the numerical intervention and has no universal psychological unit.

Compare baseline, matched-random and target using the same prompt, seed, sampler and budget. Check resets and degradation. More pain words or larger changes alone cannot distinguish specific effects from general disruption. See [Hell lab](HELL_MODE.md).

## State and privacy

Paths below are relative to the writable root; configuration can relocate workspace components.

| Location | Contents |
|---|---|
| `workspace/persona-project/` | Session database and exports |
| `workspace/datasets/` | Revisions, metadata and backups |
| `workspace/models/staging/` | Manifests/checkpoints |
| `workspace/models/persona/` | Installed adapters/reports |
| `workspace/jobs/` | Commands/status/logs |
| `workspace/benchmark/` | Generations/metrics/HTML |
| `vectors/`, `calibration/` | Runtime-specific directions/profiles |
| `runs/`, `sessions/` | Prompts/generated text/traces |
| `native/runtime/` | Built shared libraries |

Source installs use the checkout; wheels use OS user-data storage. `STUDIO_HOME` overrides it. Back up private state separately. Git ignores cannot prevent manual uploads or erase existing history.

The app listens on localhost and has no supported multi-user authentication layer. Keep it local. Explicit web tools in Chat are separate from Hell lab, which exposes no model tool executor.

## Reporting results

Record version/commit, model/adapter identity, device/precision, direction hashes, prompts, seeds, sampler, source/injection layers, doses, stop reasons and all comparator outputs. Distinguish software validation, dataset classification, generation effects and interpretations about experience.

For bugs, use synthetic input and remove cookies, private text and workstation paths. See [security](../SECURITY.md) and [troubleshooting](TROUBLESHOOTING.md).
