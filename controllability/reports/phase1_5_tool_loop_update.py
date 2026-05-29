from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    run_id = "phase1_5_tool_loop_update"
    timestamp = _utc_now()
    metrics = {
        "qwen_tool_call_template": "<tool_call>{json}</tool_call>",
        "hf_steered_verifiers_client_added": True,
        "structured_tool_call_parsing_implemented": True,
        "tool_schema_rendering_implemented": True,
        "parser_unit_tests_passed": 2,
        "swe_grep_env_import_verified": True,
        "swe_grep_tool_count": 4,
        "swe_grep_tools": ["grep", "glob_files", "read_file", "list_dir"],
        "full_tool_use_smoke_started": False,
        "phase2_scaleup_blocked": True,
        "remaining_blocker": "No GPU/Slurm runtime is available on the connected head host; srun, sinfo, and nvidia-smi are absent.",
    }
    summary = {
        "stage": "phase1_5_tool_loop_update",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": {
            "required_smoke_config": "configs/controllability/experiments/phase1_5_swe_grep_verifier_smoke_parametric_b.yaml"
        },
        "metrics": metrics,
        "gate": "blocked",
        "gate_reason": (
            "The code now supports Qwen-style structured tool calls in the HF patched-decoding client, "
            "and prime/swe-grep imports with its tool definitions. The full 8B verifier smoke was not run "
            "because the connected host has no GPU or Slurm commands available."
        ),
        "next_required_work": [
            "Run the Phase 1.5 verifier smoke from a GPU node or a working Slurm head.",
            "Verify trajectories contain executed tool calls, non-zero verifier quality scores, and agent-length token counts.",
            "Keep Phase 2 scale-up blocked until that smoke passes.",
        ],
        "next_stage_inputs": {},
    }
    run_dir = Path("runs") / run_id
    _write_json(run_dir / "summary.json", summary)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase1_5_tool_loop_update.md").write_text(
        "\n".join(
            [
                "# Phase 1.5 Tool-Loop Update",
                "",
                "The HF steered verifiers client now renders tool schemas into the Qwen chat template and parses Qwen `<tool_call>` JSON blocks into `verifiers.ToolCall` objects. This is the missing code path needed for `ToolEnv`/`StatefulToolEnv` environments to execute tool calls during patched decoding.",
                "",
                "`prime/swe-grep` now imports in the project venv and exposes four tools: `grep`, `glob_files`, `read_file`, and `list_dir`.",
                "",
                "The full 8B verifier smoke has not been run yet because the connected head host has no GPU or Slurm runtime available (`srun`, `sinfo`, and `nvidia-smi` are absent). Phase 2 remains blocked until the smoke runs on a GPU node and passes.",
                "",
            ]
        )
    )
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "phase1_5_tool_loop_update",
            "run_id": run_id,
            "gate_status": "blocked",
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": f"runs/{run_id}/summary.json",
            "prime_rl_modifications": "none",
        },
    )


if __name__ == "__main__":
    main()
