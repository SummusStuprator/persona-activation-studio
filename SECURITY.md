# Security

The application binds to localhost. Worker communication uses per-launch authentication keys over local pipes or sockets. Treat the application and job metadata directories as trusted local state.

This is a single-user local application, not a hardened multi-tenant service. Do not expose its port publicly without an independently designed authentication/security layer. A source seal detects changes relative to a local manifest; it is not publisher authentication, and anyone able to alter both source and manifest can replace it.

Persona inference loads local safetensors with remote code disabled. Training checkpoints include optimizer state and must come from a trusted training run. X session credentials remain in the configured local account database.

For a security report, request a private reporting channel through the repository maintainer. Include the affected version, operating system, entry point, and a minimal reproduction. Exclude tokens, cookies, account databases, and private model/data files.

Before sharing logs or run records, remove absolute user paths and prompt/output content. Review the complete staged diff and release archives for generated files.

The release audit uses known filename and credential patterns; a passing scan cannot prove that arbitrary prose or all possible secrets are absent. Review Git history and all branches/tags before changing visibility. `.gitignore` does not protect already tracked files. See [publishing](docs/PUBLISHING.md) and [data handling](docs/DATA_AND_PERMISSIONS.md).
