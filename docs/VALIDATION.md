# Validation

## Software

`studio check` creates isolated state, cache, model-store, and lock directories. It runs unit tests, profile/hash checks, cache recovery, recipe checks, and all 14 UI routes. The package workflow runs this command from outside the checkout after a regular wheel install.

```bash
python -m build
python -m pip install dist/BUILD.whl
studio init
studio check
studio demo
studio seal
studio verify
```

## Native runtime

Build with `studio native` or `studio native --cuda`. `studio doctor --require native` loads the libraries and reports the native identity. The native CPU CI job builds from the packaged bridge sources.

`tests/native_fleet_smoke.py --model NAME --output FILE` performs generation, intervention, and reset checks. Use `--gpu-layers N` to test an explicit offload split.

## Persona runtime

```bash
python tests/persona_fleet_smoke.py --model NAME --device CPU --output cpu.json
python tests/persona_fleet_smoke.py --model NAME --device CUDA --precision bf16 --output bf16.json
python tests/persona_fleet_smoke.py --model NAME --device CUDA --precision nf4 --output nf4.json
```

These tests require the selected local adapter. They verify generation, storage-precision injection error, and reset logits. The output records the actual device, precision, ABI, and source hashes.

## Training and benchmark

`tests/optimizer_resume_smoke.py` compares an interrupted tiny LoRA run with an uninterrupted run. `tests/qlora_gpu_smoke.py` exercises the NF4 loader and production trainer on CUDA. `studio demo --train` creates a local fixture for the benchmark.

Run `studio benchmark --model-root PATH --only MODEL --output-root RESULTS` to produce baseline and adapted generations, held-out NLL, and reports. A second invocation with the same identity resumes the result set. Changed sampler/data/model/code inputs require a new output directory.

`tests/ui_persona_cuda_smoke.py` exercises the actual sidebar load action. `tests/persona_direction_smoke.py` fits a direction, reloads its bank, and validates CUDA NF4 generation under that control. Both accept `--model NAME --output FILE`.
