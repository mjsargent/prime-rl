# Phase 1.5 Tool-Loop Update

The HF steered verifiers client now renders tool schemas into the Qwen chat template and parses Qwen `<tool_call>` JSON blocks into `verifiers.ToolCall` objects. This is the missing code path needed for `ToolEnv`/`StatefulToolEnv` environments to execute tool calls during patched decoding.

`prime/swe-grep` now imports in the project venv and exposes four tools: `grep`, `glob_files`, `read_file`, and `list_dir`.

The full 8B verifier smoke has not been run yet because the connected head host has no GPU or Slurm runtime available (`srun`, `sinfo`, and `nvidia-smi` are absent). Phase 2 remains blocked until the smoke runs on a GPU node and passes.
