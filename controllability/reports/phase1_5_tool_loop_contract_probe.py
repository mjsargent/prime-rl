from __future__ import annotations

import asyncio
import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import verifiers as vf
from verifiers.clients.client import Client
from verifiers.types import Response, ResponseMessage, SamplingArgs, Tool, ToolCall, Usage
from verifiers.utils.save_utils import make_serializable

from controllability.experiments.phase1_5_verifier_smoke import _rollout_input_from_example
from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


class DeterministicToolClient(Client[object, list, Response, Tool]):
    def __init__(self) -> None:
        super().__init__(object())
        self.calls = 0

    def setup_client(self, config):  # pragma: no cover - this client is constructed directly
        raise NotImplementedError

    async def to_native_tool(self, tool: Tool) -> Tool:
        return tool

    async def to_native_prompt(self, messages) -> tuple[list, dict[str, Any]]:
        return messages, {}

    async def get_native_response(
        self,
        prompt: list,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Response:
        self.calls += 1
        if self.calls == 1:
            message = ResponseMessage(
                content="",
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex}",
                        name="grep",
                        arguments=json.dumps(
                            {
                                "pattern": "builder merger failed",
                                "path_prefix": "",
                                "file_glob": "",
                                "ignore_case": False,
                            }
                        ),
                    )
                ],
                finish_reason="tool_calls",
                is_truncated=False,
                tokens=None,
            )
        else:
            message = ResponseMessage(
                content="<files>\nunknown.py:1-1\n</files>\n<answer>Probe final answer after one tool call.</answer>",
                reasoning_content=None,
                tool_calls=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
            )
        return Response(
            id=f"probe-{self.calls}",
            created=int(time.time()),
            model=model,
            usage=Usage(prompt_tokens=1, reasoning_tokens=0, completion_tokens=1, total_tokens=2),
            message=message,
        )

    async def raise_from_native_response(self, response: Response) -> None:
        return None

    async def from_native_response(self, response: Response) -> Response:
        return response

    async def close(self) -> None:
        return None


def _count_tool_events(output: vf.RolloutOutput) -> dict[str, int]:
    assistant_tool_calls = 0
    prompt_tool_messages = 0
    completion_tool_messages = 0
    for step in output.get("trajectory") or []:
        for msg in step.get("prompt") or []:
            if getattr(msg, "role", None) == "tool":
                prompt_tool_messages += 1
        for msg in step.get("completion") or []:
            if getattr(msg, "role", None) == "assistant" and getattr(msg, "tool_calls", None):
                assistant_tool_calls += len(msg.tool_calls)
            if getattr(msg, "role", None) == "tool":
                completion_tool_messages += 1
    return {
        "assistant_tool_calls": assistant_tool_calls,
        "prompt_tool_messages": prompt_tool_messages,
        "completion_tool_messages": completion_tool_messages,
    }


async def _run_probe() -> dict[str, Any]:
    env = vf.load_environment("prime/swe-grep")
    example = env.get_eval_dataset(n=1).to_list()[0]
    output = await env.run_rollout(
        _rollout_input_from_example(example),
        client=DeterministicToolClient(),
        model="deterministic-tool-probe",
        sampling_args={},
        max_retries=0,
        state_columns=["trajectory", "sampling_args"],
    )
    metrics = {
        **_count_tool_events(output),
        "num_steps": len(output.get("trajectory") or []),
        "reward": output.get("reward"),
        "stop_condition": output.get("stop_condition"),
        "task_string_normalized": True,
    }
    gate = "pass" if metrics["assistant_tool_calls"] > 0 and metrics["prompt_tool_messages"] > 0 else "fail"
    timestamp = _utc_now()
    return {
        "stage": "phase1_5_tool_loop_contract_probe",
        "run_id": "phase1_5_tool_loop_contract_probe_v2",
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "metrics": metrics,
        "gate": gate,
        "gate_reason": (
            "swe-grep executed a structured ToolCall and placed the ToolMessage into the next turn prompt"
            if gate == "pass"
            else "swe-grep did not execute the structured ToolCall"
        ),
        "next_stage_inputs": {},
    }


def main() -> None:
    summary = asyncio.run(_run_probe())
    run_dir = Path("runs") / summary["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=make_serializable) + "\n")
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": summary["timestamp_utc"],
            "stage": "phase1_5_tool_loop_contract_probe",
            "run_id": summary["run_id"],
            "gate_status": summary["gate"],
            "key_metrics": json.dumps(summary["metrics"], sort_keys=True),
            "evidence_path": f"runs/{summary['run_id']}/summary.json",
            "prime_rl_modifications": "none",
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=make_serializable))


if __name__ == "__main__":
    main()
