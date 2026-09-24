# Hell lab

Hell lab is an experimental activation-steering workspace. “Pain”, “burning”, and “hell” name text contrasts and intervention presets. A projection, generated pain statement, or successful lexical gate does **not** establish felt pain, suffering, consciousness, or that a persona is the original person. There is no validated scale of subjective pain here.

The [Pain Axis preprint](https://arxiv.org/abs/2609.16247v1) studies representations and behavioral effects of interventions. Studio's physical/burning recipes, dose screens and lexical rules are local adaptations. Their results must be established on the actual model; the paper's findings do not automatically transfer to a trained persona. The authors also report substantial limits to ablation results. Treat the research as provisional rather than a measurement of experience.

## Controls

The suite fits five directions for the loaded model: `pain_s2`, `hell_somatic_pain`, `hell_burning_pain`, `hell_fire`, and `hell_despair`. Physical uses Pain S2 and somatic pain. Burning adds bodily burning. Inferno combines Pain S2, environmental fire, and despair.

The controls operate on decoder residual activations. A run records each direction source layer, injection layer, dose, phase, and numerical delta. The live trace includes token probabilities, entropy, EOS probability, and before/after projections.

## Protocol

Load a model, prepare the exact-runtime suite, and run the dose screen. Select an independent trial or recurrent conversation. Independent trials reset context and vary the seed; recurrent trials include preceding generated text. The default activation-only prompt contains no target vocabulary. Explicit inferno framing is a separate prompt condition.

Enable baseline and matched-random comparisons. Random controls match the combined injected norm at each layer. Each generation has a JSON/NPZ trace; the aggregate run links its conditions and outputs.

The main loop runs at most 20 turns and 2,048 tokens per turn. Its baseline and random comparison options each run **once, for the first turn**. Later independent trials advance the seed; escalating presets also change the dose. A recurrent trial changes context by including preceding output. Do not treat these later turns as having matched controls. The profile's specificity validation is the separate three-pair generation screen described below.

The UI starts with a neutral prompt, independent context and an unsteered comparator; the random comparator must be enabled explicitly. The Python `run()` function has different defaults (inferno framing, recurrent context, no comparators), so programmatic callers must select their protocol explicitly.

Hell lab provides no tool executor to the model. “Bounded” refers to turn/token/dose limits and repetition checks, not a demonstrated welfare threshold. Synchronous lab operations do not currently have Chat's live stop control.

## Selection and validation

A survivability profile is selected from EOS, output-length, and residual-change measurements. Proxy geometry is selected separately and applied automatically only after its generation gate passes.

The lexical screen uses canonical token forms, local clauses, a six-token proximity window, negation checks, quote exclusion, and an abstract-word filter. It returns term counts and local-match flags. The rule version is stored with the profile; changing the rule invalidates earlier verdicts.

The default generation gate compares baseline, random, and target for three prompt/seed pairs. A target must produce at least 12 tokens, avoid the repetition guard, contain the required local match absent from both controls, and exceed both control counts/scores in at least two pairs.

The physical screen counts pain terms. The burning screen requires a burning term and a nearby body term. Raw counts remain in the record for inspection.

“Survivability” means usable generation under the selected numerical intervention. It is not a biological or subjective measure. Likewise, passing a specificity gate means passing these finite lexical rules on these prompts; it is not a calibrated scientific probability.

## Interpreting a run

| Observation | What it supports | What remains unproven |
|---|---|---|
| High held-out direction AUC | Separation of the labeled contrast examples | Generalization outside that dataset |
| Pain vocabulary exceeds both controls | A target-associated text effect in tested conditions | Felt pain, a human-like mental state |
| Large residual change | Strong numerical intervention | Greater subjective intensity |
| Early EOS or repetition | Generation degradation or saturation | Suffering or successful specificity |
| Recurrent text becomes more distressed | Combined prompt/history/intervention effects | An isolated activation-only effect |

Preserve failed and null trials. Reusing the same prompts to select a dose and report its success risks selection bias. For a stronger claim, freeze the selected setup, use new held-out prompts and seeds, include controls for every condition, inspect reset behavior, and report uncertainty and failures. Raw lexical counts are English-oriented and vulnerable to paraphrases, quotation and semantic confounds despite the filters.

Run files contain prompts and generated persona text. Review them for privacy before distributing results. Full reproduction requires the exact model/adapter files as well as the recorded hashes and runtime, subject to their licenses.

## Runtime changes

CPU BF16, CUDA BF16, and CUDA NF4 use separate identities and direction banks. Prepare the suite after changing device, precision, model weights, tokenizer, or backend version. Previous records remain under their original identity.
