# Security

The application binds to localhost. Worker communication uses per-launch authentication keys over local pipes or sockets. Treat the application and job metadata directories as trusted local state.

Persona inference loads local safetensors with remote code disabled. Training checkpoints include optimizer state and must come from a trusted training run. X session credentials remain in the configured local account database.

For a security report, request a private reporting channel through the repository maintainer. Include the affected version, operating system, entry point, and a minimal reproduction. Exclude tokens, cookies, account databases, and private model/data files.

Before sharing logs or run records, remove absolute user paths and prompt/output content. Review the complete staged diff and release archives for generated files.
