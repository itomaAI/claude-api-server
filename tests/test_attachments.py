"""添付の扱い: claude へコンテンツブロックのまま渡す(ファイルへ落とさない・Read を使わない)。"""

import base64
import os
import sys
from dataclasses import dataclass
from typing import Any, Union

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conversation as cv  # noqa: E402
from conversation import (  # noqa: E402
    SYSTEM_ATTACHMENT_NOTE,
    Attachment,
    Turn,
    build_plan,
    conversation_key,
    decode_data_uri,
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


def uri(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def image_part(raw: bytes = b"\x89PNG one", mime: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": uri(mime, raw)}}


def file_part(name: str, mime: str, raw: bytes) -> dict:
    return {"type": "file", "file": {"filename": name, "file_data": uri(mime, raw)}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


# --------------------------------------------------------------------------
# data URI
# --------------------------------------------------------------------------

def test_decode_data_uri():
    assert decode_data_uri(uri("image/PNG", b"abc")) == ("image/png", b"abc")
    assert decode_data_uri("https://example.com/x.png") is None
    assert decode_data_uri("data:text/plain,not-base64") is None
    assert decode_data_uri("") is None


# --------------------------------------------------------------------------
# 種類ごとの扱い
# --------------------------------------------------------------------------

def test_image_becomes_an_image_block():
    raw = b"\x89PNG one"
    (part,) = render_parts([image_part(raw)])
    assert isinstance(part, Attachment)
    assert part.kind == "image" and part.media_type == "image/png"
    assert base64.b64decode(part.data) == raw
    assert part.block() == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(raw).decode(),
        },
    }


def test_image_sent_as_file_part_is_still_an_image_block():
    (part,) = render_parts([file_part("shot.jpg", "image/jpeg", b"\xff\xd8 jpeg")])
    assert isinstance(part, Attachment) and part.kind == "image"


def test_pdf_becomes_a_document_block_with_its_name_in_the_text():
    turn = Turn.from_parts(
        "user", render_parts([file_part("invoice.pdf", "application/pdf", b"%PDF-1.7")])
    )
    blocks = render_blocks([turn])
    assert [b["type"] for b in blocks] == ["text", "document"]
    assert "invoice.pdf" in blocks[0]["text"]
    assert blocks[1]["source"]["media_type"] == "application/pdf"


@pytest.mark.parametrize(
    "name,mime,raw",
    [
        ("a.txt", "text/plain", "文字のファイル".encode()),
        ("a.json", "application/json", b'{"k": 1}'),
        ("a.svg", "image/svg+xml", b"<svg xmlns='http://www.w3.org/2000/svg'/>"),
        ("a.csv", "application/octet-stream", b"a,b\n1,2\n"),
    ],
)
def test_text_like_files_are_inlined(name, mime, raw):
    (part,) = render_parts([file_part(name, mime, raw)])
    assert isinstance(part, str)
    assert raw.decode() in part
    assert name in part


def test_binary_of_unsupported_type_is_omitted_with_a_note():
    (part,) = render_parts([file_part("a.zip", "application/zip", b"PK\x03\x04\x00\xff")])
    assert isinstance(part, str)
    assert "omitted" in part and "a.zip" in part


def test_no_path_and_no_read_tool_in_anything_sent():
    parts = render_parts([text_part("see"), image_part(), text_part("ok")])
    turn = Turn.from_parts("user", parts)
    assert "Read tool" not in turn.text
    assert "/attachments/" not in turn.text


def test_system_attachments_are_replaced_by_a_note():
    convo = render_messages([Msg("system", [text_part("S"), image_part()]), Msg("user", "hi")])
    assert convo.system_prompt == "S" + SYSTEM_ATTACHMENT_NOTE


# --------------------------------------------------------------------------
# 送る中身(ブロック)
# --------------------------------------------------------------------------

def test_blocks_keep_the_position_of_the_image():
    turn = Turn.from_parts(
        "user", render_parts([text_part("before"), image_part(), text_part("after")])
    )
    blocks = render_blocks([turn])
    assert [b["type"] for b in blocks] == ["text", "image", "text"]
    assert blocks[0]["text"] == "before" and blocks[2]["text"] == "after"


def test_whitespace_only_text_between_attachments_is_dropped():
    turn = Turn.from_parts(
        "user",
        render_parts([image_part(b"one"), text_part("\n"), image_part(b"two"), text_part("  ")]),
    )
    assert [b["type"] for b in render_blocks([turn])] == ["image", "image"]


def test_full_history_is_flattened_with_images_in_place():
    turns = [
        Turn.from_parts("user", render_parts([text_part("what is this?"), image_part()])),
        Turn("assistant", "a cat"),
        Turn("user", "and now?"),
    ]
    blocks = render_blocks(turns)
    assert [b["type"] for b in blocks] == ["text", "image", "text"]
    assert blocks[0]["text"] == "User: what is this?"
    assert blocks[2]["text"] == "\n\nAssistant: a cat\n\nUser: and now?"
    # 文字での姿では、画像は印になっている
    assert "[Attached image: image/png sha256:" in render_turns(turns)


# --------------------------------------------------------------------------
# 会話キー: 同じ添付は同じキー、違う添付は違うキー
# --------------------------------------------------------------------------

def _turns(raw: bytes) -> list[Turn]:
    return [Turn.from_parts("user", render_parts([text_part("look"), image_part(raw)]))]


def test_same_attachment_gives_the_same_key():
    assert conversation_key("s", None, None, _turns(b"one")) == conversation_key(
        "s", None, None, _turns(b"one")
    )


def test_different_attachment_gives_a_different_key():
    assert conversation_key("s", None, None, _turns(b"one")) != conversation_key(
        "s", None, None, _turns(b"two")
    )


def test_key_does_not_contain_the_payload():
    big = b"x" * 100_000
    (turn,) = _turns(big)
    assert len(turn.text) < 200


# --------------------------------------------------------------------------
# 実行プラン
# --------------------------------------------------------------------------

@pytest.fixture
def store(monkeypatch):
    fresh = SessionStore(max_items=8, ttl=3600)
    monkeypatch.setattr(cv, "STORE", fresh)
    monkeypatch.setattr(cv, "SESSION_REUSE", True)
    monkeypatch.setattr(cv, "STATS", {"hit": 0, "miss": 0, "resume_failed": 0})
    return fresh


def test_resume_sends_only_the_new_turn_with_its_image(store):
    first = [Msg("system", "S"), Msg("user", [text_part("hello")])]
    build_plan(render_messages(first), "sonnet", None).register("sess-1", "world")

    followup = first + [
        Msg("assistant", "world"),
        Msg("user", [text_part("tool result"), image_part()]),
    ]
    plan = build_plan(render_messages(followup), "sonnet", None)

    assert plan.resume_id == "sess-1"
    assert [b["type"] for b in plan.blocks] == ["text", "image"]
    assert plan.blocks[0]["text"] == "tool result"
    assert plan.has_attachment is True


def test_conversation_with_an_image_in_the_past_still_resumes(store):
    """過去のターンに画像があっても、次のターンで引き当たる(キーが安定している)。"""
    first = [Msg("user", [text_part("look"), image_part()])]
    build_plan(render_messages(first), None, None).register("sess-1", "a cat")

    followup = first + [Msg("assistant", "a cat"), Msg("user", "thanks")]
    plan = build_plan(render_messages(followup), None, None)

    assert plan.resume_id == "sess-1"
    assert plan.blocks == [{"type": "text", "text": "thanks"}]
    assert plan.has_attachment is False


def test_fallback_to_full_resends_the_images(store):
    first = [Msg("user", [text_part("look"), image_part()])]
    build_plan(render_messages(first), None, None).register("sess-1", "a cat")
    followup = first + [Msg("assistant", "a cat"), Msg("user", "thanks")]
    plan = build_plan(render_messages(followup), None, None)

    plan.fallback_to_full()

    assert plan.resume_id is None
    assert [b["type"] for b in plan.blocks] == ["text", "image", "text"]
    assert plan.has_attachment is True
