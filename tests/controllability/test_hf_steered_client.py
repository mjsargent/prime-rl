from __future__ import annotations

import json

from controllability.envs.hf_steered_client import _parse_tool_calls_from_text


def test_parse_qwen_tool_call_blocks():
    content, tool_calls = _parse_tool_calls_from_text(
        'I should inspect files.\n<tool_call>\n{"name":"grep","arguments":{"pattern":"foo","path":"src"}}\n</tool_call>'
    )

    assert content == "I should inspect files."
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "grep"
    assert json.loads(tool_calls[0].arguments) == {"pattern": "foo", "path": "src"}


def test_parse_malformed_tool_call_leaves_text_unchanged():
    raw = '<tool_call>\n{"name": "grep", "arguments": \n</tool_call>'
    content, tool_calls = _parse_tool_calls_from_text(raw)

    assert content == raw
    assert tool_calls == []
