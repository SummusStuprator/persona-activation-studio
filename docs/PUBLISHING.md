# Release procedure

The version is defined in `studio_version.py`. Package metadata reads the same value.

Before tagging, run the software, installed-wheel, native, and ML integration checks in [VALIDATION.md](VALIDATION.md). Record results against the release commit. The Actions workflow covers Windows/Linux package installation and a Linux CPU-native build.

Build an sdist and wheel. Inspect both archives for model files, authentication databases, account exports, generated arrays, logs, and absolute workstation paths. Scan the intended Git history, not only the current tree.

Merge the reviewed release commit into the branch being published. Tag that exact commit. Publish the tested source archive and wheel with SHA-256 checksums. Repository visibility is a separate maintainer action.

Generated state is excluded by `.gitignore`; model caches and user datasets are external inputs. `paper/datasets/` contains attributed research fixtures with a separate license.

The local `trusted-files.json` seal is machine state and is not committed. Recreate it after installation with `studio seal`.
