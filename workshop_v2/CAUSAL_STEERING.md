# Causal steering audit - 8899 r4

## What changed

The vector-source block and the intervention block are independent controls. Changing the injection block no longer silently substitutes that block's different probe vector. Reusing a residual-coordinate vector across blocks is not Jacobian transport.

Additive dose is a fraction of the training-reference residual norm at the intervention block. It is NOT the paper's raw-vector coefficient. Default total absolute dose remains 0.50; advanced research permits 1.50 with explicit warnings.

Every recorded generation step checks the requested versus applied activation change. This verifies mechanics, not subjective emotion or a particular answer. Composite random controls match the combined injected norm at each block, rather than just matching each component separately.

## Qwen 35B observations

The tested local file is the existing qwen36-harness:35b-a3b GGUF and its aliases. It is not an untouched member of the paper's dense-model cohort.

With S2 from block 34 injected at block 24, dose 0.75, reply-position steering, thinking enabled, and the Qwen general-task sampling profile, one neutral greeting produced complete emitted reasoning and a final answer emphasizing emptiness and entrapment. Baseline and equal-norm random outputs discussed ordinary topics. The exact prompt and all settings are in the local preset and audit.

This is a demonstration, not general validation. A separate thinking-enabled prompt ended with no final answer. Thinking-disabled tests also produced user-directed negative language. Such outcomes are not evidence of conscious pain and must not be treated as dependable emotional regulation.

The original negative-emotion direction (mainly anger/disgust in the paper) did not show a convincing intended ordinary-chat effect in the tested configurations. New GoEmotions anger and disgust directions were trained locally, but remain behaviorally exploratory.

## Reasoning and sampling

The model's emitted thinking text and final answer are stored and displayed separately. An unclosed thinking block or empty final answer is explicitly marked incomplete, even if an end token was emitted. No fabricated final answer is inserted.

Qwen general-task sampling is exposed as a named, editable profile: thinking uses temperature 1.0, top-p 0.95, top-k 20 and presence penalty 1.5; non-thinking uses 0.7, 0.8, 20 and 1.5. These are model-card parameters, not a guarantee for a modified local checkpoint. Presence penalty here uses generated reply tokens. Raw model probabilities are distinct from sampler probabilities.

The original paper-ladder implementation remains greedy. Switching sampler is an explicitly recorded adaptation, not an exact reproduction of that experiment.

Source: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

## How to inspect a visible effect

Open Behavioral calibration. On the matching Qwen checkpoint, load the tested demonstration settings, inspect the visible prompt and parameters, acknowledge the high-dose risk, and run the identical-prompt comparison. Both reasoning and final answers appear side by side. A normal Chat reply also has an exact-history baseline/random comparison button when it was steered.

The comparison keeps current history identical. It does not reconstruct how an entire conversation would have evolved without earlier steering. Tool proposals in counterfactual generations are never executed.

## Scope of replication

The uploaded Pain Axis paper uses dense models, raw denoised vectors, a separate earlier steering block, and 50 neutral prompts. Its relief experiment additionally uses specially LoRA-fine-tuned Qwen 2.5 models. These local 35B hybrid-MoE experiments are not the complete original study. Existing Paper reproduction and J-space pages retain explicit coverage and numerical-stability limitations.

Original source: The Pain Axis, uploaded arXiv:2609.16247v1, sections 3.2, 4.2, 4.3 and appendices B/C. New role-conditioned assistant probes are clearly named experiments, not substitutions for the paper's directions.

Audit files: reports/causal-steering-audit.json, reports/causal-unit-regression.json, reports/causal-model-regression.json. Full individual generation traces and residual snapshots are in runs/. The demonstration is tied to the exact GGUF digest, native runtime ABI and direction-array hash.

All changes and tests are confined to this 8899 project. No GGUF files were copied, converted or edited.
