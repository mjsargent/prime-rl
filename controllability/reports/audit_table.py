from __future__ import annotations

from pathlib import Path

HEADER = "| timestamp_utc | stage | run_id | gate_status | key_metrics | evidence_path | prime_rl_modifications |\n|---|---|---|---|---|---|---|\n"


def append_audit_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER)
    with path.open("a") as f:
        f.write(
            "| {timestamp_utc} | {stage} | {run_id} | {gate_status} | {key_metrics} | {evidence_path} | {prime_rl_modifications} |\n".format(
                **row
            )
        )
