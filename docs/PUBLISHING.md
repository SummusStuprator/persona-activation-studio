# Publishing checklist (maintainer-facing)

This file records the steps used to publish Persona Activation Studio publicly.
The local build process never configures a Git remote for you.

## State of this tree

- `v0.1.0` — first validated portable release (clean clone, fresh venv, native build, smoke tests, UI health, secret audit; see docs/VALIDATION.md).
- `v0.2.0` — Physical/Burning pain modules, model-launch lease serialization, and the pair-cache/physical-profile/recipe-quality tests; full suite green (native llama3.2:1b, persona cutyourtrachea_qwen3_1.7b_lora_v2 CPU, 14 UI pages).
- Release identity: `studio seal` / `studio verify` (trusted-files.json is local-only and Git-ignored).
- Tracked files carry no machine paths, session cookies, credentials, model weights, datasets, or generated artifacts; `.gitignore` is designed around that separation.

## Steps

1. **License** — MIT (LICENSE). Paper-derived material keeps its own license (paper/LICENSE, THIRD_PARTY.md).

2. **Create the remote and push**

   ```bash
   gh repo create SummusStuprator/persona-activation-studio --private   # or --public
   git remote add origin https://github.com/SummusStuprator/persona-activation-studio.git
   git push -u origin main
   git push origin --tags
   ```

3. **Repository metadata (suggested)**

   - Description: `Local-first persona studio: collect X data → build datasets → train persona adapters → inspect and steer them at the activation level (Hell lab).`
   - Topics: `llm`, `interpretability`, `activation-steering`, `peft`, `lora`, `llama.cpp`, `streamlit`.

4. **Post-push audit** (expect empty output)

   ```bash
   git ls-files | grep -iE '\.gguf|\.safetensors|\.venv/|trusted-files\.json|^studio\.toml$|persona-sources\.json|resource-policy\.json|\.key$|\.token$'
   ```

5. **Docs links** — confirm README install snippets match the published repository URL.

## Optional follow-ups

- **CI**: a GitHub Actions workflow (ubuntu + windows) running `bash -n scripts/*.sh`, `studio doctor`, and the model-free tests: `tests/physical_recipe_quality.py`, `tests/physical_profile_smoke.py`, `tests/pair_cache_smoke.py`.
- **Release notes**: reuse docs/VALIDATION.md per tag.
