# Temporary Qwen2.5-0.5B GGUF relay for NEXUS R2

This branch is a temporary public GitHub Actions relay because private-repository runners for `Shtenco/agi_olga` currently fail before allocating a machine.

The workflow downloads only the official file:

- repository: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- filename: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- required SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`

The model is not committed to Git. It is stored for three days as a workflow artifact. The checksum evidence is stored separately for thirty days. This relay should remain unmerged and may be deleted after the NEXUS R2 integration test is complete.
