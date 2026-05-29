# Phase 1.5 Verifier Integration Blocker

Phase 2 scale-up remains blocked.

The HF steered verifiers client has been added and can produce patched-decoding assistant text inside the verifiers rollout loop. It does not yet emit structured tool calls. For swe-grep, that means the current client cannot produce full tool-using trajectories or verifier-derived quality scores.

The 20-prompt Phase 1.5 smoke was not started because it would fail the known tool-call requirement rather than test the fixed pipeline.

Next implementation step: either parse Qwen tool-call text into `AssistantMessage.tool_calls`, or add a prime-rl/vLLM residual-steering hook that keeps the existing prime-rl tool-call path intact.
