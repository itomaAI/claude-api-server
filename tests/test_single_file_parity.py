"""単一ファイル版 (claude_api_server.py) とモジュール版の挙動が一致することを検証する。

両者は独立に保守されるため、ロジックが乖離したらここで落ちる。
"""

import base64
import os
import sys
from dataclasses import dataclass
from typing import Any, Union

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import claude as mod_claude  # noqa: E402
import claude_api_server as single  # noqa: E402
import conversation as mod_conv  # noqa: E402
from sessions import SessionStore  # noqa: E402


@dataclass
class Msg:
    role: str
    content: Union[str, list[dict[str, Any]]]


def user(text: str) -> Msg:
    return Msg("user", [{"type": "text", "text": text}])


CONVERSATIONS = [
    [Msg("system", "S"), user("hello")],
    [Msg("system", "S"), user("hello"), Msg("assistant", "world"), user("again")],
    [user("no system prompt")],
    [Msg("system", "A"), Msg("system", "B"), user("two system messages")],
]


@pytest.mark.parametrize("messages", CONVERSATIONS)
def test_render_messages_parity(messages):
    a = mod_conv.render_messages(messages)
    b = single.render_messages(messages)
    assert a.system_prompt == b.system_prompt
    assert a.system_has_attachment == b.system_has_attachment
    assert [(t.role, t.text) for t in a.turns] == [(t.role, t.text) for t in b.turns]


@pytest.mark.parametrize("messages", CONVERSATIONS)
def test_render_turns_parity(messages):
    a = mod_conv.render_messages(messages).turns
    b = single.render_messages(messages).turns
    assert mod_conv.render_turns(a) == single.render_turns(b)


@pytest.mark.parametrize("messages", CONVERSATIONS)
def test_conversation_key_parity(messages):
    a = mod_conv.render_messages(messages)
    b = single.render_messages(messages)
    assert mod_conv.conversation_key("S", "sonnet", "high", a.turns) == \
        single.conversation_key("S", "sonnet", "high", b.turns)
    assert mod_conv.conversation_root_key("S", None, None, a.turns) == \
        single.conversation_root_key("S", None, None, b.turns)


def test_invalid_conversation_parity():
    bad = [user("hi"), Msg("assistant", "trailing assistant")]
    with pytest.raises(mod_conv.InvalidConversation):
        mod_conv.render_messages(bad)
    with pytest.raises(single.InvalidConversation):
        single.render_messages(bad)


def test_attachment_path_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(mod_conv, "ATTACH_DIR", str(tmp_path))
    monkeypatch.setattr(single, "ATTACH_DIR", str(tmp_path))
    uri = "data:text/plain;base64," + base64.b64encode(b"same bytes").decode()
    part = [{"type": "file", "file": {"filename": "a.txt", "file_data": uri}}]

    assert mod_conv.render_content(part) == single.render_content(part)


@pytest.fixture
def isolated_stores(monkeypatch):
    """両実装に独立した空のストアを差し込む。"""
    for module in (mod_conv, single):
        monkeypatch.setattr(module, "STORE", SessionStore(max_items=8, ttl=3600))
        monkeypatch.setattr(module, "STATS", {"hit": 0, "miss": 0, "resume_failed": 0})
        monkeypatch.setattr(module, "SESSION_REUSE", True)


def _plan_pair(messages, model="sonnet", effort=None):
    a = mod_conv.build_plan(mod_conv.render_messages(messages), model, effort)
    b = single.build_plan(single.render_messages(messages), model, effort)
    return a, b


def test_build_plan_parity_first_turn(isolated_stores):
    a, b = _plan_pair(CONVERSATIONS[0])
    assert (a.resume_id, a.prompt) == (b.resume_id, b.prompt)


def test_build_plan_parity_after_register(isolated_stores):
    first = CONVERSATIONS[0]
    a, b = _plan_pair(first)
    a.register("sess-1", "world")
    b.register("sess-1", "world")

    a2, b2 = _plan_pair(CONVERSATIONS[1])
    assert a2.resume_id == b2.resume_id == "sess-1"
    assert a2.prompt == b2.prompt == "again"


def test_build_plan_parity_tampered_reply(isolated_stores):
    a, b = _plan_pair(CONVERSATIONS[0])
    a.register("sess-1", "world")
    b.register("sess-1", "world")

    tampered = [
        Msg("system", "S"),
        user("hello"),
        Msg("assistant", "world [edited]"),
        user("again"),
    ]
    a2, b2 = _plan_pair(tampered)
    assert a2.resume_id is None and b2.resume_id is None
    assert a2.prompt == b2.prompt


def test_fallback_to_full_parity(isolated_stores):
    a, b = _plan_pair(CONVERSATIONS[0])
    a.register("sess-1", "world")
    b.register("sess-1", "world")

    a2, b2 = _plan_pair(CONVERSATIONS[1])
    a2.fallback_to_full()
    b2.fallback_to_full()
    assert a2.prompt == b2.prompt
    assert a2.resume_id is None and b2.resume_id is None


@pytest.mark.parametrize(
    "resume_id,has_attachment,model,effort",
    [
        (None, False, None, None),
        (None, True, "sonnet", None),
        ("sess-1", False, "opus", "high"),
        ("sess-1", True, None, "low"),
    ],
)
def test_build_command_parity(resume_id, has_attachment, model, effort):
    kwargs = dict(
        prompt="p",
        system_prompt="s",
        has_attachment=has_attachment,
        turns=[],
        model=model,
        effort=effort,
        resume_id=resume_id,
    )
    a = mod_claude.build_command(mod_conv.Plan(**kwargs), "/tmp/sys.txt")
    b = single.build_command(single.Plan(**kwargs), "/tmp/sys.txt")
    assert a == b


def test_usage_conversion_parity():
    usage = {
        "input_tokens": 2,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 300,
        "output_tokens": 7,
    }
    import server as mod_server

    assert mod_server.to_openai_usage(usage) == single.to_openai_usage(usage)


def test_effective_effort_resolution():
    """reasoning_effort (OpenAI 互換名) と effort の解決が両版で一致する。"""
    import claude_api_server as single
    import server as modular

    cases = [
        ({"effort": "high"}, "high"),
        ({"reasoning_effort": "high"}, "high"),
        ({"reasoning_effort": "minimal"}, "low"),          # claude に minimal は無い
        ({"effort": "max", "reasoning_effort": "low"}, "max"),  # effort 優先
        ({}, None),
    ]
    msgs = [{"role": "user", "content": "hi"}]
    for kwargs, expected in cases:
        for mod in (single, modular):
            req = mod.ChatRequest(messages=msgs, **kwargs)
            assert req.effective_effort == expected, (mod.__name__, kwargs)
