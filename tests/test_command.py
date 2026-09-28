"""claude の起動引数と、stdin へ渡す中身。"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import claude  # noqa: E402
from conversation import Plan  # noqa: E402

IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
}


def plan(**kwargs) -> Plan:
    base = dict(prompt="p", system_prompt="s", turns=[])
    base.update(kwargs)
    return Plan(**base)


def _value_after(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_no_tools_and_no_mcp_servers_even_with_an_attachment():
    for p in (plan(), plan(content=[IMAGE_BLOCK])):
        cmd = claude.build_command(p, None)
        assert _value_after(cmd, "--tools") == ""
        assert "--strict-mcp-config" in cmd
        assert "--mcp-config" not in cmd
        assert "Read" not in cmd
        assert "--permission-mode" not in cmd


def test_input_and_output_are_stream_json():
    cmd = claude.build_command(plan(), None)
    assert _value_after(cmd, "--input-format") == "stream-json"
    assert _value_after(cmd, "--output-format") == "stream-json"
    assert "--verbose" in cmd


def test_resume_model_effort_and_system_prompt_are_passed():
    cmd = claude.build_command(
        plan(resume_id="sess-1", model="opus", effort="high"), "/tmp/sys.txt"
    )
    assert _value_after(cmd, "--resume") == "sess-1"
    assert _value_after(cmd, "--model") == "opus"
    assert _value_after(cmd, "--effort") == "high"
    assert _value_after(cmd, "--system-prompt-file") == "/tmp/sys.txt"


def test_payload_is_one_json_line_with_a_user_message():
    payload = claude.stdin_payload(plan(prompt="こんにちは"))
    assert payload.endswith(b"\n") and payload.count(b"\n") == 1
    assert json.loads(payload) == {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "こんにちは"}],
        },
    }


def test_payload_with_newlines_in_the_text_is_still_one_line():
    payload = claude.stdin_payload(plan(prompt="a\nb\n\nc"))
    assert payload.count(b"\n") == 1
    assert json.loads(payload)["message"]["content"][0]["text"] == "a\nb\n\nc"


def test_payload_carries_the_blocks_as_they_are():
    blocks = [{"type": "text", "text": "look"}, IMAGE_BLOCK]
    payload = claude.stdin_payload(plan(content=blocks))
    assert json.loads(payload)["message"]["content"] == blocks


def test_parse_result_event_picks_the_result():
    out = (
        b'{"type":"system","subtype":"init","session_id":"s"}\n'
        b'garbage line\n'
        b'\n'
        b'{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
        b'{"type":"result","is_error":false,"result":"hi","session_id":"s","usage":{"output_tokens":1}}\n'
    )
    event = claude.parse_result_event(out)
    assert event["result"] == "hi" and event["session_id"] == "s"


def test_parse_result_event_returns_none_without_a_result():
    assert claude.parse_result_event(b'{"type":"system"}\n') is None
    assert claude.parse_result_event(b"") is None
