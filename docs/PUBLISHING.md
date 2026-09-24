# Publishing a release

Use Git to upload source. Attach the wheel and source distribution to a GitHub Release. Do not upload the working directory through the browser.

Use a versioned release branch and publish only after that commit's release checks pass. Checks for an earlier version do not certify a new release.

## 1. Prepare the source

Set the version in `studio_version.py` and `CITATION.cff`, then add its entry to `CHANGELOG.md`. Use a new version for changes to an already tagged build. Do not replace published tags or assets.

```powershell
git status --short
git diff --check
python -m studio_cli check
git diff --stat
```

Review the staged file names and changes before committing. Model weights, adapters, account databases, exports, configuration, vectors, logs, and runs remain outside Git. `paper/datasets` contains attributed fixtures.

## 2. Build and validate

Build from a clean committed checkout. The release audit requires both distributions to match that checkout.

```powershell
python -m pip install build
python -m build
python scripts/verify_release.py --write
python scripts/verify_release.py
python scripts/verify_release.py --history
```

The audit writes `dist/SHA256SUMS.txt` and `dist/release.json`. The latter binds both artifacts to the source commit and tree. Keep previous versions under their original names.

To retain existing release metadata, pass `--outdir dist/release-0.3.4` to the build and `--dist dist/release-0.3.4` to the audit; use that directory for all following asset paths. A candidate tested before a source change is stale and must be rebuilt. Both audit modes require a clean checkout except the separate history scan. Pattern scans complement manual review; they do not prove absence of every secret or private text fragment.

## 3. Push a release branch and review it

For this release, use `release/0.3.4` and tag `v0.3.4`.

```powershell
$Repo = 'SummusStuprator/persona-activation-studio'
git push -u origin release/0.3.4
gh pr create --repo $Repo --base main --head release/0.3.4 --title 'Release 0.3.4' --body-file docs/releases/0.3.4.md
gh run list --repo $Repo --branch release/0.3.4
gh pr checks --repo $Repo --watch
```

Reuse an existing release pull request instead of creating a duplicate. Require the `Release gate` check. It aggregates all six Windows/Linux Python package jobs and the Linux CPU-native job. CUDA results are recorded separately under `docs/READINESS.md`.

The workflow runs on pushes to `release/**` and pull requests against `main`. Wait for the release commit's gate before merging or publishing.

Merge with a merge commit, preserving the tested release commit. Then fetch and fast-forward local `main`. Do not use force push, `git push --all`, or `git push --tags`.

## 4. Tag the tested commit

Read the commit from `dist/release.json`. Reuse an existing matching tag; a mismatched tag is an error.

```powershell
$Release = Get-Content dist/release.json -Raw | ConvertFrom-Json
git rev-parse HEAD
git tag -a v0.3.4 $Release.commit -m 'Persona Activation Studio 0.3.4'
git push origin refs/tags/v0.3.4
```

The tag must resolve to `$Release.commit`, and that commit must be reachable from `origin/main` after merge. The workflow's tested artifacts are retained for 14 days under `release-assets-COMMIT`; download them before the retention period expires when using CI-built files.

## 5. Upload a draft release

Use the files matching the validated manifest, not `dist/*`.

```powershell
gh release create v0.3.4 --repo $Repo --verify-tag --draft --title 'Persona Activation Studio 0.3.4' --notes-file docs/releases/0.3.4.md dist/persona_activation_studio-0.3.4-py3-none-any.whl dist/persona_activation_studio-0.3.4.tar.gz dist/SHA256SUMS.txt dist/release.json
gh release view v0.3.4 --repo $Repo --web
```

Check the tag, source commit, notes, and four attachments. Download the attachments to a new folder and compare their SHA-256 hashes with `SHA256SUMS.txt`. Do not use `--clobber` to replace a published asset.

## 6. Publish

Repository visibility and release publication are separate operations. To make the source public, use repository **Settings → General → Danger Zone → Change repository visibility** after reviewing all branches and history being exposed.

Before changing visibility, refresh remote refs with `git fetch origin --prune --tags`, rerun `python scripts/verify_release.py --history`, and inspect all branches/tags plus repository issues, pull requests and release attachments that will become public. The local history scan does not inspect remote-only refs or GitHub discussion content. It does not authorize history rewriting; resolve any finding before proceeding.

Publish the reviewed draft:

```powershell
gh release edit v0.3.4 --repo $Repo --draft=false --latest
```

Confirm the public repository and release in a signed-out browser. Install the downloaded wheel in a new virtual environment, run `studio init` and `studio check`, then remove the temporary environment after saving the output.

## Repository settings

Protect `main` with pull requests and the required `Release gate`; block force pushes and deletion. Keep workflow permissions read-only. Use GitHub-hosted runners for untrusted pull requests. Pin actions to full commit SHAs; Dependabot proposes weekly action updates. Enable dependency alerts, secret scanning, push protection, and private vulnerability reporting where available. Restrict tag updates and enable immutable releases after all draft assets are attached.

Do not commit generated experiment data or ML checkpoints to this source repository. Publish datasets and adapters separately with their licenses and evaluation metadata. Keep PyPI publication separate; use Trusted Publishing rather than a long-lived upload token if PyPI distribution is added.
