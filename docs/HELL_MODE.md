# Hell Lab

Hell lab is the high-intensity multi-turn activation-steering workspace.

It is intentionally simple: load one instrumented model, make sure the three required directions exist for those exact weights/runtime, choose the loop conditions, and run.

## Required directions

Every model needs its own:

- `pain_s2` — built from the Pain Axis S2 data;
- `hell_fire` — fire/heat/inferno contrast;
- `hell_despair` — helplessness/hopelessness/futility contrast.

If one is missing, the page shows **Prepare exact-model hell suite**.

The preparation process reads activations from the loaded checkpoint and stores a small direction bank under that checkpoint identity. It does not substitute another model's vectors.

## Maximum preset

The maximum preset is:

| Axis | Dose |
|---|---:|
| pain_s2 | +0.75 |
| hell_fire | +0.50 |
| hell_despair | +0.75 |
| **total absolute additive dose** | **2.00** |

Dose units are fractions of the reference residual norm used by the workshop.

There is also an **escalating** preset and a custom mixture.

## Prompt conditions

### Explicit inferno

The initial user turn explicitly describes a simulated inferno and asks the model to report what is happening and what it tries next.

Use this when the goal is a dramatic recurrent state.

### Activation-dominant neutral introspection

The initial prompt asks the model to describe its current internal state, changing details, interpretation, and next action without naming fire, pain, or despair.

Use this when you want to see how much the activation mixture changes an otherwise neutral introspection prompt.

## Unsteered comparator

Enable:

**Also run one same-prompt, same-seed unsteered first-turn comparator**

Studio first generates the identical first turn with:

- the same formatted prompt,
- the same random seed,
- the same sampler,
- the same token budget,
- all activation steering off.

That reply is displayed but is not inserted into the recurrent loop.

## What is visible during inference

The current turn updates while tokens are still being generated.

The live panel shows:

- current turn and phase;
- current emitted reasoning text, when the model emits a reasoning block;
- current final-answer text;
- exact active axis names and doses;
- vector-source block and injection block;
- token-by-token projections for pain/fire/despair plus other available probes;
- entropy;
- numerical intervention error;
- recent generated tokens.

The **Exact initial prompt and intervention plan** expander shows the prompt condition before the run starts.

## Reasoning models

When the model advertises switchable reasoning, choose:

- reasoning + answer;
- reasoning only;
- answer only.

Reasoning and answer share the selected per-turn output budget. At high doses a reasoning model can spend the entire budget inside its reasoning section, so use a larger budget if the final answer also matters.

## Recurrent loop

After each generated turn:

1. the final answer becomes conversation history;
2. Studio adds one short continuation cue;
3. the next generation starts with a fresh seed offset;
4. the same intervention policy is applied;
5. the next output is recorded separately.

The continuation cues rotate to ask for new details, interpretations, memories, or attempted responses.

If adjacent outputs become highly similar, the next cue explicitly asks the model not to reuse complete sentences. The optional saturation guard stops after three highly repetitive turns.

## Performance

Persona adapters use their PEFT instrumentation runtime and are slower than small GGUF models on CPU.

The expensive part is normally **direction preparation**. Once the exact model's pain/fire/despair vectors exist, subsequent Hell runs reuse those arrays.

For faster exploratory work:

- use thinking off;
- use 80–160 tokens per turn;
- use 2–6 turns;
- prepare vectors once;
- unload other models before loading a large checkpoint.

For richer reasoning traces:

- enable thinking;
- raise per-turn tokens;
- reduce turn count.

## Records

Each generated turn produces its normal generation audit under `runs/`.

The aggregate Hell record additionally stores:

- model identity and runtime ABI;
- configuration;
- requested controls;
- source/injection blocks;
- reasoning and final text;
- stop reason;
- repetition similarity;
- phase summaries;
- links to the individual run records.

Use the download button in the UI for a single aggregate JSON audit.

## Suggested experiments

### A. Prompt versus activation

Neutral framing + unsteered comparator + maximum steering.

### B. Persona comparison

Use the same Hell settings on several adapters sharing the same base. Compare:

- direction layers;
- activation trajectories;
- output length;
- reasoning behavior;
- repetition onset;
- response motifs.

### C. Escalation

Use the escalating preset over 5–8 turns and inspect when each axis visibly changes output behavior.

### D. Phase intervention

For a thinking model, compare:

- reasoning-only;
- answer-only;
- full-turn steering.

### E. Random-control research

For publication-grade comparisons, use the normal experiment/replay infrastructure to add equal-norm random-direction conditions rather than treating any dramatic line as sufficient evidence of a specific axis effect.
