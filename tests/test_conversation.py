"""claude を起動しない純粋ロジックのテスト。

    python3 -m pytest tests -q
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Union

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conversation as cv  # noqa: E402
from conversation import (  # noqa: E402
    InvalidConversation,
    Turn,
    build_plan,
    conversation_key,
    conversation_root_key,
    parts_text,
    render_blocks,
    render_messages,
    render_parts,
    render_turns,
)
from sessions import SessionStore  # noqa: E402


@dataclass
class Msg:
    role: str
    content: Union[str, list[dict[str, Any]]]


def user(text: str) -> Msg:
    """Itera など OpenAI 互換クライアントが送る content-parts 形式。"""
    return Msg("user", [{"type": "text", "text": text}])


# --------------------------------------------------------------------------
# レンダリング
# --------------------------------------------------------------------------

def test_plain_string_content():
    assert render_parts("hello") == ["hello"]


def test_content_parts_are_concatenated():
    parts = render_parts([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert parts_text(parts) == "ab"
    assert all(isinstance(p, str) for p in parts)


def test_non_data_uri_image_is_flagged_not_attached():
    parts = render_parts(
        [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]
    )
    assert all(isinstance(p, str) for p in parts)
    assert "omitted" in parts_text(parts)


def test_system_messages_are_collected_separately():
    convo = render_messages([Msg("system", "S1"), user("hi"), Msg("system", "S2")])
    assert convo.system_prompt == "S1\n\nS2"
    assert [t.role for t in convo.turns] == ["user"]


def test_conversation_must_end_with_user():
    with pytest.raises(InvalidConversation):
        render_messages([user("hi"), Msg("assistant", "yo")])
    with pytest.raises(InvalidConversation):
        render_messages([Msg("system", "only system")])


def test_single_user_turn_is_rendered_bare():
    assert render_turns([Turn("user", "hi")]) == "hi"


def test_multi_turn_rendering_labels_speakers():
    rendered = render_turns([Turn("user", "a"), Turn("assistant", "b"), Turn("user", "c")])
    assert rendered == "User: a\n\nAssistant: b\n\nUser: c"


def test_text_only_blocks_are_one_text_block_equal_to_the_rendered_text():
    """文字だけの会話では、送る中身は render_turns と同じ文字列の text ブロック1つ。"""
    for turns in (
        [Turn("user", "hi")],
        [Turn("user", "a"), Turn("assistant", "b"), Turn("user", "c")],
    ):
        assert render_blocks(turns) == [{"type": "text", "text": render_turns(turns)}]


# --------------------------------------------------------------------------
# 会話キー
# --------------------------------------------------------------------------

def test_key_changes_with_any_component():
    turns = [Turn("user", "hi")]
    base = conversation_key("sys", "sonnet", None, turns)
    assert base != conversation_key("other", "sonnet", None, turns)
    assert base != conversation_key("sys", "opus", None, turns)
    assert base != conversation_key("sys", "sonnet", "high", turns)
    assert base != conversation_key("sys", "sonnet", None, [Turn("user", "hi!")])


def test_root_key_is_stable_as_conversation_grows():
    first = [Turn("user", "hi")]
    grown = first + [Turn("assistant", "yo"), Turn("user", "again")]
    assert conversation_root_key("s", None, None, first) == conversation_root_key(
        "s", None, None, grown
    )


# --------------------------------------------------------------------------
# セッション引き当て
# --------------------------------------------------------------------------

@pytest.fixture
def store(monkeypatch):
    fresh = SessionStore(max_items=8, ttl=3600)
    monkeypatch.setattr(cv, "STORE", fresh)
    monkeypatch.setattr(cv, "SESSION_REUSE", True)
    monkeypatch.setattr(cv, "STATS", {"hit": 0, "miss": 0, "resume_failed": 0})
    return fresh


def test_first_turn_misses_and_sends_everything(store):
    convo = render_messages([Msg("system", "S"), user("hello")])
    plan = build_plan(convo, "sonnet", None)
    assert plan.resume_id is None
    assert plan.prompt == "hello"
    assert plan.blocks == [{"type": "text", "text": "hello"}]
    assert cv.STATS["miss"] == 1


def test_next_turn_resumes_and_sends_only_the_delta(store):
    convo = render_messages([Msg("system", "S"), user("hello")])
    plan = build_plan(convo, "sonnet", None)
    plan.register("sess-1", "world")

    followup = render_messages(
        [Msg("system", "S"), user("hello"), Msg("assistant", "world"), user("again")]
    )
    plan2 = build_plan(followup, "sonnet", None)

    assert plan2.resume_id == "sess-1"
    assert plan2.prompt == "again", "差分ターンのみを送る"
    assert plan2.blocks == [{"type": "text", "text": "again"}]
    assert cv.STATS["hit"] == 1


def test_tampered_reply_falls_back_to_full_send(store):
    """クライアントが応答を加工して保存していたら引き当てず、全履歴を送る。"""
    convo = render_messages([Msg("system", "S"), user("hello")])
    build_plan(convo, "sonnet", None).register("sess-1", "world")

    followup = render_messages(
        [
            Msg("system", "S"),
            user("hello"),
            Msg("assistant", "world [edited by client]"),
            user("again"),
        ]
    )
    plan = build_plan(followup, "sonnet", None)

    assert plan.resume_id is None
    assert "hello" in plan.prompt and "again" in plan.prompt


def test_different_model_does_not_reuse_session(store):
    convo = render_messages([Msg("system", "S"), user("hello")])
    build_plan(convo, "sonnet", None).register("sess-1", "world")

    followup = render_messages(
        [Msg("system", "S"), user("hello"), Msg("assistant", "world"), user("again")]
    )
    assert build_plan(followup, "opus", None).resume_id is None


def test_fallback_to_full_restores_whole_history(store):
    convo = render_messages([Msg("system", "S"), user("hello")])
    build_plan(convo, "sonnet", None).register("sess-1", "world")

    followup = render_messages(
        [Msg("system", "S"), user("hello"), Msg("assistant", "world"), user("again")]
    )
    plan = build_plan(followup, "sonnet", None)
    assert plan.resume_id == "sess-1"

    plan.fallback_to_full()
    assert plan.resume_id is None
    assert plan.prompt == "User: hello\n\nAssistant: world\n\nUser: again"
    assert plan.blocks == [{"type": "text", "text": plan.prompt}]


def test_session_reuse_disabled_always_sends_everything(store, monkeypatch):
    monkeypatch.setattr(cv, "SESSION_REUSE", False)
    convo = render_messages([Msg("system", "S"), user("hello")])
    plan = build_plan(convo, "sonnet", None)
    plan.register("sess-1", "world")

    followup = render_messages(
        [Msg("system", "S"), user("hello"), Msg("assistant", "world"), user("again")]
    )
    assert build_plan(followup, "sonnet", None).resume_id is None
