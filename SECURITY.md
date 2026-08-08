# Security Policy

## Generated code is untrusted by default

AI Language Pro can generate and execute Python scaffolds. Model output and generated artifacts must be treated as untrusted input unless they were reviewed or produced entirely from trusted deterministic source.

`ai-language run` uses Python isolated mode (`-I`) and a wall-clock timeout, but **it is not an operating-system sandbox**. Python code can still access files, processes and network resources available to the current user.

For untrusted workloads use a disposable container/VM with:

- no host secrets mounted;
- no cloud instance credentials;
- read-only filesystem where practical;
- restricted network egress;
- CPU/memory/process limits;
- a non-privileged user;
- disposable working data.

## API keys

Do not commit `OPENAI_API_KEY` or `.env` files. Prefer environment variables or a secrets manager. The CLI never needs an API key for deterministic `generate`, `inspect`, `check` or `run` operations.

## Model output validation

The `plan` command validates model-proposed DSL with the deterministic parser before compilation. Future structured planners must preserve this fail-closed behavior.

## Reporting

For a security issue, avoid publishing exploit details in a public issue before a fix is available. Contact the repository owner through GitHub first and coordinate disclosure.
