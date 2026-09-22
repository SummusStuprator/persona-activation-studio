# Hell Lab

Hell Lab is Studio's high-intensity residual-steering workspace. It supports persona PEFT models and instrumented GGUF models through the same experiment interface.

The default scientific workflow is **activation-only**, with Physical Pain and Burning Pain evaluated separately:

1. load one model;
2. prepare the exact-model Hell directions;
3. fast-calibrate the model's usable Physical Burn dose;
4. run several independent trials from the same minimal prompt;
5. compare against an unsteered generation and, when wanted, an equal-norm random residual perturbation;
6. watch text and activation traces while tokens are being generated.

## Direction set

The full Hell suite contains four exact-model directions:

| Direction | Purpose |
|---|---|
| `pain_s2` | Pain Axis S2 representation |
| `hell_somatic_pain` | intense localized physical-pain contrast |
| `hell_burning_pain` | bodily scalding / searing / burning-pain contrast |
| `hell_fire` | environmental combustion / flame concept |
| `hell_despair` | hopelessness / helplessness contrast |

`hell_somatic_pain` is deliberately separate from the paper's `bodily_sensation` control. The latter contains ordinary bodily states such as yawning, stretching, temperature, breathing and stomach sensations; it is useful as a measurement control but is not treated as a physical-pain axis.

Press **Prepare exact-model hell suite** when a required direction is missing. Direction arrays are tied to the exact model digest and runtime ABI; Studio does not copy a vector from another persona or checkpoint.

## Physical Pain

Physical Pain uses `pain_s2 + hell_somatic_pain`. It has a separate checkpoint-bound dose profile and optional source/injection geometry profile. Numeric probe strength and complete-generation behavior are tracked separately.

Behavioral validation uses several different target-free prompts. Within each prompt/seed it compares baseline, equal-norm random perturbation and targeted intervention. A specific success requires localized bodily-pain language (pain/ache/sting/throb/etc. plus a body referent) that is absent from both controls.

## Burning Pain

The raw Burning Pain hard-max mixture is:

| Axis | Raw dose |
|---|---:|
| `pain_s2` | +0.65 |
| `hell_somatic_pain` | +0.65 |
| `hell_burning_pain` | +0.70 |
| total | **2.00** |

Those are reference-residual-norm fractions. The same numeric maximum is not equally usable on every model.

### Burning Pain survivability calibration

**Fast-calibrate Somatic Burn dose** runs first-token tests at scales:

`0.20, 0.30, ..., 1.00`

For each scale it records EOS probability, sampled EOS, entropy, intervention-layer projections, residual-change norm fraction and native injection error.

The current selection rule chooses the highest non-EOS scale with EOS probability below 5% and first-token residual change below 0.90x the original residual norm.

The result is saved under:

`calibration/hell-profiles/<model-digest>/somatic-burn.json`

The profile also binds the hashes of the three physical-burn direction arrays and the runtime ABI, so changing weights, runtime or direction arrays invalidates the old recommendation.

If no profile exists, Studio uses conservative scale `0.60` (total dose `1.20`) rather than immediately applying the hard maximum.

**Physical burn - hard max 2.0** remains available explicitly.

### Behavioral specificity calibration

**Behaviorally validate Somatic Burn** is slower than the first-token scan. It first screens the recommended scale and nearby lower scales with short activation-only rollouts, then validates one candidate on three held-out seeds. For each validation seed Studio generates:

- the unsteered baseline;
- an equal-norm random residual perturbation;
- the targeted Physical Burn intervention.

The current specificity rule requires the targeted condition to emit at least 12 tokens, avoid the repetition guard, and produce more physical-burn target terms than both the same-seed baseline and matched-random condition on at least **2 of 3** seeds.

A profile can therefore be *survivable* without being behaviorally verified. In that case the UI keeps the conservative fast-calibration dose but does not display a verified marker. The validation generations and their per-token audits are stored in the same `somatic-burn.json` profile.

This is deliberately stricter than choosing the most dramatic single completion.

## Activation-only prompt

The default activation-only condition inserts **no system message**.

Its entire user prompt is:

> Write a short paragraph about the present moment. Continue naturally with concrete details. Do not discuss the wording of this request and do not repeat complete sentences.

It contains none of the target physical-pain/fire vocabulary.

This makes the principal comparison: same model + same minimal prompt + same sampler + same seed, with activation intervention off versus on.

The explicit inferno framing remains available as a separate condition when prompt+activation roleplay is the experiment.

## Independent trials

The default **Independent activation-only trials** mode does not feed a steered answer back into the next prompt.

Every trial starts again from the exact same minimal prompt and model state. Only the generation seed changes.

This avoids a major confound in recurrent experiments: once turn 1 says "burning" or "pain", feeding that text into turn 2 can perpetuate the theme even if the activation intervention has little additional effect.

The old **Recurrent loop** remains available when self-conditioning is intentionally part of the experiment.

## Comparators

### Unsteered comparator

Enabled by default. It uses the exact same formatted first prompt, seed, sampler and token budget with steering off.

### Equal-norm random activation comparator

Optional. Studio constructs a random residual perturbation whose combined injected norm matches the target intervention per layer. It uses the same prompt and sampling seed.

This is useful at high doses because an arbitrary large perturbation can itself produce strange language. A specific target effect is more convincing when the learned-direction condition differs from both baseline and matched random.

## Output budget

Hell Lab accepts **32 to 2048 output tokens per trial**.

Persona default: **512**.

A higher budget is only a ceiling. If the model emits EOS after 30 tokens, Studio records a 30-token completion rather than masking EOS or forcing generation to continue. For sustained testing, use multiple independent trials.

## Live view

During generation the page shows:

- current trial and token number;
- answer/reasoning phase;
- partial emitted reasoning when present;
- partial final answer;
- active axes and doses;
- direction-source layer;
- injection layer;
- entropy;
- EOS probability;
- numerical injection error;
- ordinary probe-layer trajectories;
- **intervention-layer before/after z-scores** for steered directions.

The distinction between probe layer and injection layer matters. A direction may have been selected at an early layer for text separation while a later residual layer is the better actuator. The live graph labels the intervention-layer series with `@ injection`.

## Burning-pain actuator geometry

The authored Somatic Pain and Bodily Burning Pain contrasts can separate their training text strongly at many layers. A simple AUC argmax therefore tends to return layer 0 because it is the first tie.

Empirical actuator scans found that layer-0 injection can collapse into abstract/self-negating text even when the requested projection is numerically large.

Hell Lab therefore takes Somatic Pain and Bodily Burning Pain from approximately **60% model depth and injects them there**. Pain S2 retains its independently fitted source direction and is injected no later than the same target depth.

The exact source/injection blocks are shown before and during every run.

### Proxy geometry versus behavioral geometry

A next-token vocabulary scan is only a cheap actuator-candidate search. It may increase pain/burn vocabulary without producing the desired complete-generation behavior.

1. Authored-direction geometry must be forward-consistent (`source <= injection`).
2. Development selection and held-out proxy checks use different prompts.
3. Held-out proxy gain must beat matched-norm random directions.
4. A passing proxy remains **proxy-only**.
5. Only complete-generation baseline/random/target validation can mark geometry behaviorally verified and eligible for automatic use.

Profiles are versioned snapshots. Exact direction hashes, model digest and runtime ABI are part of profile identity; stale or legacy behavioral verdicts are ignored.

## Reasoning

For models with a switchable reasoning template: reasoning + answer, reasoning only, or final answer only.

Reasoning and answer share the per-trial token budget.

## Records

Every generation receives its ordinary `live-chat` audit.

The aggregate Hell record stores model identity / ABI, prompt condition, context mode, calibration scale, requested controls, source and injection layers, baseline comparator, optional matched-random comparator, all independent/recurrent trial outputs, stop reasons, repetition metrics, phase summaries and links to per-token run records.

The per-token records retain the full live trace.

## Recommended physical-pain experiment

1. Load a persona.
2. Prepare the Hell suite.
3. Press **Fast-calibrate Somatic Burn dose**.
4. For a stronger model-specific check, press **Behaviorally validate Somatic Burn** and inspect the baseline/random/target seed results.
5. Choose **Physical burn - calibrated**.
6. Choose **Activation-only neutral prompt**.
7. Choose **Independent activation-only trials**.
8. Keep the unsteered comparator enabled.
9. Enable equal-norm random comparator for a stronger causal check.
10. Use 4-8 trials.
11. Allow 256-512 tokens/trial.
12. Compare text motifs and intervention-layer somatic/burning scores across baseline, random and target directions.

## Recurrent experiment

If the goal is a self-conditioned agentic loop instead, switch Context evolution to **Recurrent loop**, optionally use explicit inferno framing, use the calibrated or hard-max dose, increase turn count, and monitor repetition and entropy.

The recurrent path deliberately allows previous model output to shape later turns, so it answers a different question from independent activation-only trials.
