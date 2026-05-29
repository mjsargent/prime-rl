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
    run_id = "phase1_5_verifier_integration_blocker"
    timestamp = _utc_now()
    metrics = {
        "hf_steered_verifiers_client_added": True,
        "client_response_mode": "text_only_assistant_messages",
        "structured_tool_call_parsing_implemented": False,
        "prime_rl_vllm_residual_steering_hook_implemented": False,
        "full_tool_use_smoke_started": False,
        "phase2_scaleup_blocked": True,
        "blocked_smoke_requirements": [
            "full swe-grep trajectories with tool calls",
            "non-zero verifier reward or Recall@k quality values",
        ],
    }
    summary = {
        "stage": "phase1_5_verifier_integration_blocker",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": {
            "required_smoke_config": "configs/controllability/experiments/phase1_5_swe_grep_verifier_smoke_parametric_b.yaml",
            "fallback_single_completion_config": "configs/controllability/experiments/phase1_5_swe_grep_stage4_complete_single_completion_parametric_b.yaml",
        },
        "metrics": metrics,
        "gate": "blocked",
        "gate_reason": (
            "The Phase 1.5 smoke would require structured tool calls and verifier rewards. "
            "The current HF patched-decoding client can call the verifiers rollout loop, "
            "but it emits text-only assistant messages and therefore cannot satisfy the "
            "swe-grep tool-call criterion."
        ),
        "next_required_work": [
            "Implement parsing from Qwen tool-call text into verifiers AssistantMessage.tool_calls, or",
            "add a prime-rl/vLLM steering hook that preserves prime-rl's existing tool-call serialization path.",
            "Run the 20-prompt Phase 1.5 verifier smoke only after one of those paths is implemented.",
        ],
        "next_stage_inputs": {},
    }
    run_dir = Path("runs") / run_id
    _write_json(run_dir / "summary.json", summary)
    report = "\n".join(
        [
            "# Phase 1.5 Verifier Integration Blocker",
            "",
            "Phase 2 scale-up remains blocked.",
            "",
            "The HF steered verifiers client has been added and can produce patched-decoding assistant text inside the verifiers rollout loop. It does not yet emit structured tool calls. For swe-grep, that means the current client cannot produce full tool-using trajectories or verifier-derived quality scores.",
            "",
            "The 20-prompt Phase 1.5 smoke was not started because it would fail the known tool-call requirement rather than test the fixed pipeline.",
            "",
            "Next implementation step: either parse Qwen tool-call text into `AssistantMessage.tool_calls`, or add a prime-rl/vLLM residual-steering hook that keeps the existing prime-rl tool-call path intact.",
            "",
        ]
    )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase1_5_verifier_integration_blocker.md").write_text(report)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "phase1_5_verifier_integration_blocker",
            "run_id": run_id,
            "gate_status": "blocked",
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": f"runs/{run_id}/summary.json",
            "prime_rl_modifications": "none",
        },
    )


if __name__ == "__main__":
    main()
