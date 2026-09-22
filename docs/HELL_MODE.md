# Hell lab

## Controls

The suite fits five directions for the loaded model: `pain_s2`, `hell_somatic_pain`, `hell_burning_pain`, `hell_fire`, and `hell_despair`. Physical uses Pain S2 and somatic pain. Burning adds bodily burning. Inferno combines Pain S2, environmental fire, and despair.

The controls operate on decoder residual activations. A run records each direction source layer, injection layer, dose, phase, and numerical delta. The live trace includes token probabilities, entropy, EOS probability, and before/after projections.

## Protocol

Load a model, prepare the exact-runtime suite, and run the dose screen. Select an independent trial or recurrent conversation. Independent trials reset context and vary the seed; recurrent trials include preceding generated text. The default activation-only prompt contains no target vocabulary. Explicit inferno framing is a separate prompt condition.

Enable baseline and matched-random comparisons. Random controls match the combined injected norm at each layer. Each generation has a JSON/NPZ trace; the aggregate run links its conditions and outputs.

## Selection and validation

A survivability profile is selected from EOS, output-length, and residual-change measurements. Proxy geometry is selected separately and applied automatically only after its generation gate passes.

The lexical screen uses canonical token forms, local clauses, a six-token proximity window, negation checks, quote exclusion, and an abstract-word filter. It returns term counts and local-match flags. The rule version is stored with the profile; changing the rule invalidates earlier verdicts.

The default generation gate compares baseline, random, and target for three prompt/seed pairs. A target must produce at least 12 tokens, avoid the repetition guard, contain the required local match absent from both controls, and exceed both control counts/scores in at least two pairs.

The physical screen counts pain terms. The burning screen requires a burning term and a nearby body term. Raw counts remain in the record for inspection.

## Runtime changes

CPU BF16, CUDA BF16, and CUDA NF4 use separate identities and direction banks. Prepare the suite after changing device, precision, model weights, tokenizer, or backend version. Previous records remain under their original identity.
