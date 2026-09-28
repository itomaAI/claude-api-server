"""OpenAI 形式のメッセージ列を claude へ渡す形に変換し、
セッションを引き当てて実行プランを決めるところまで。
"""

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, Union

from config import LOOKBACK_TURNS, SESSION_REUSE
from sessions import STATS, STORE

DATA_URI_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)

Content = Union[str, list[dict[str, Any]]]


class RawMessage(Protocol):
    """OpenAI 形式の1メッセージ(pydantic モデルでもテスト用の簡易オブジェクトでも可)。"""

    role: str
    content: Content


class InvalidConversation(ValueError):
    """クライアントの messages が受け付けられない形だった。"""


# --------------------------------------------------------------------------
# 添付
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Attachment:
    """claude へコンテンツブロックのまま渡す添付(画像・PDF)。

    以前は data URI をディスクへ落とし、Read ツールで読ませる案内文に置き換えていた。
    claude -p は --input-format stream-json で画像・文書のブロックを直接受け取れる
    (2.1.280 で確認)ので、ツールを介さずそのまま渡す。
    """

    kind: str  # "image" | "document"
    media_type: str
    data: str  # base64
    digest: str  # デコードした中身の sha256(先頭32桁)
    filename: str = ""

    def marker(self) -> str:
        """文字での姿。会話キーと記録に使う。

        中身ではなくハッシュで表すので、同じ添付は常に同じ印になり
        会話ハッシュが安定する(= セッション再利用が効く)。
        """
        if self.kind == "image":
            return f"\n[Attached image: {self.media_type} sha256:{self.digest}]\n"
        return (
            f"\n[Attached file '{self.filename}': "
            f"{self.media_type} sha256:{self.digest}]\n"
        )

    def block(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data,
            },
        }


# ターンの中身の1片。文字か、ブロックのまま渡す添付。
Part = Union[str, Attachment]


def decode_data_uri(data_uri: str) -> Optional[tuple[str, bytes]]:
    """base64 の data: URI を (MIME, 中身) にする。そうでなければ None。"""
    m = DATA_URI_RE.match(data_uri or "")
    if not m or not m.group(2):
        return None
    mime = (m.group(1) or "application/octet-stream").split(";")[0].strip().lower()
    try:
        raw = base64.b64decode(m.group(3))
    except (binascii.Error, ValueError):
        return None
    return mime, raw


def _as_text(raw: bytes) -> Optional[str]:
    """UTF-8 の文字として読めるならその文字列。バイナリなら None。"""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return None if "\0" in text else text


def attachment_part(mime: str, raw: bytes, filename: str = "") -> Part:
    """添付1つを、claude へ渡せる形にする。

    - 画像(SVG を除く) -> image ブロック。大きさ・形式の調整は claude 側がやる
      (9.7MB の JPEG、幅 9000px の PNG、BMP が通ることを確認済み)。
    - PDF -> document ブロック。
    - 文字として読めるもの(SVG を含む) -> 本文へそのまま埋め込む。
    - それ以外 -> 送らず、その旨の注記にする。
    """
    digest = hashlib.sha256(raw).hexdigest()[:32]
    data = base64.b64encode(raw).decode("ascii")

    if mime.startswith("image/") and mime != "image/svg+xml":
        return Attachment("image", mime, data, digest, filename)
    if mime == "application/pdf":
        return Attachment("document", mime, data, digest, filename)

    text = _as_text(raw)
    name = filename or "file"
    if text is not None:
        return f"\n[Attached file '{name}']\n{text}\n[End of attached file '{name}']\n"
    return f"[file attachment omitted: '{name}' ({mime}) is not a supported type]"


def render_parts(content: Content) -> list[Part]:
    """OpenAI 形式の content を、文字と添付の並びにする。"""
    if isinstance(content, str):
        return [content]

    parts: list[Part] = []
    for part in content:
        ptype = part.get("type")
        if ptype == "text":
            parts.append(part.get("text", ""))
        elif ptype == "image_url":
            decoded = decode_data_uri((part.get("image_url") or {}).get("url", ""))
            if decoded:
                parts.append(attachment_part(*decoded))
            else:
                parts.append("[image attachment omitted: not a data URI]")
        elif ptype == "file":
            file_obj = part.get("file") or {}
            filename = file_obj.get("filename", "file")
            decoded = decode_data_uri(file_obj.get("file_data", ""))
            if decoded:
                parts.append(attachment_part(*decoded, filename=filename))
            else:
                parts.append(f"[file attachment omitted: {filename} is not a data URI]")
    return parts


def parts_text(parts: Sequence[Part]) -> str:
    """文字での姿。添付は印(Attachment.marker)で表す。"""
    return "".join(p if isinstance(p, str) else p.marker() for p in parts)


# --------------------------------------------------------------------------
# メッセージ -> ターン
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    text: str  # 文字での姿(会話キー・記録用)。添付は印で入っている
    parts: tuple[Part, ...] = ()  # 実際に送る中身。空なら text だけのターン

    @classmethod
    def from_parts(cls, role: str, parts: Sequence[Part]) -> "Turn":
        return cls(role, parts_text(parts), tuple(parts))

    @property
    def pieces(self) -> tuple[Part, ...]:
        return self.parts or (self.text,)

    @property
    def has_attachment(self) -> bool:
        return any(isinstance(p, Attachment) for p in self.parts)


@dataclass(frozen=True)
class Conversation:
    turns: list[Turn]
    system_prompt: str


SYSTEM_ATTACHMENT_NOTE = (
    "[attachment omitted: attachments in system messages are not supported]"
)


def render_messages(messages: Sequence[RawMessage]) -> Conversation:
    system_texts: list[str] = []
    turns: list[Turn] = []

    for m in messages:
        parts = render_parts(m.content)
        if m.role == "system":
            # system prompt はファイルで渡す文字列なので、ブロックを載せられない。
            system_texts.append(
                "".join(
                    p if isinstance(p, str) else SYSTEM_ATTACHMENT_NOTE for p in parts
                )
            )
        else:
            turns.append(Turn.from_parts(m.role, parts))

    if not turns or turns[-1].role != "user":
        raise InvalidConversation("messages must end with a user message")

    return Conversation(turns, "\n\n".join(system_texts))


def _speaker(turn: Turn) -> str:
    return "User" if turn.role == "user" else "Assistant"


def render_turns(turns: Sequence[Turn]) -> str:
    """ターン列の文字での姿(記録・試験用)。実際に送るのは render_blocks の結果。"""
    if len(turns) == 1 and turns[0].role == "user":
        return turns[0].text
    return "\n\n".join(f"{_speaker(t)}: {t.text}" for t in turns)


def render_blocks(turns: Sequence[Turn]) -> list[dict[str, Any]]:
    """ターン列を、claude に渡す1つの user メッセージのコンテンツブロック列にする。

    文字だけの会話なら、render_turns と同じ文字列の text ブロック1つになる。
    添付はその位置にブロックとして挟まる。
    """
    pieces: list[Part] = []
    if len(turns) == 1 and turns[0].role == "user":
        pieces.extend(turns[0].pieces)
    else:
        for i, t in enumerate(turns):
            pieces.append(("\n\n" if i else "") + f"{_speaker(t)}: ")
            pieces.extend(t.pieces)

    blocks: list[dict[str, Any]] = []
    buf: list[str] = []

    def flush() -> None:
        text = "".join(buf)
        buf.clear()
        # 空白だけの text ブロックは API が受け付けないので作らない。
        if text.strip():
            blocks.append({"type": "text", "text": text})

    for p in pieces:
        if isinstance(p, str):
            buf.append(p)
            continue
        if p.kind == "document":
            buf.append(f"\n[Attached file '{p.filename or 'file'}']\n")
        flush()
        blocks.append(p.block())
    flush()

    return blocks or [{"type": "text", "text": render_turns(turns)}]


# --------------------------------------------------------------------------
# セッション引き当て用のキー
# --------------------------------------------------------------------------

def conversation_key(
    system_prompt: str,
    model: Optional[str],
    effort: Optional[str],
    turns: Sequence[Turn],
) -> str:
    payload = json.dumps(
        {
            "system": system_prompt,
            "model": model or "",
            "effort": effort or "",
            "turns": [[t.role, t.text] for t in turns],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def conversation_root_key(
    system_prompt: str,
    model: Optional[str],
    effort: Optional[str],
    turns: Sequence[Turn],
) -> str:
    """会話が何ターン進んでも変わらない識別子(直列化の単位)。

    先頭ターンで会話を識別する。クライアントが古いターンを切り落とした場合は
    別の会話とみなされるが、その場合は引き当ても外れてフル送信になるので安全側。
    """
    return conversation_key(system_prompt, model, effort, turns[:1])


# --------------------------------------------------------------------------
# 実行プラン
# --------------------------------------------------------------------------

@dataclass
class Plan:
    """1リクエストをどう claude に投げるかの決定。"""

    prompt: str  # 送る中身の文字での姿(記録・試験用)。添付は印で表す
    system_prompt: str
    turns: list[Turn]
    # 実際に送るコンテンツブロック。空なら prompt を text ブロック1つとして送る。
    content: list[dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None
    effort: Optional[str] = None
    resume_id: Optional[str] = None

    @property
    def blocks(self) -> list[dict[str, Any]]:
        return self.content or [{"type": "text", "text": self.prompt}]

    @property
    def has_attachment(self) -> bool:
        return any(b.get("type") != "text" for b in self.content)

    def register(self, session_id: Optional[str], reply: str) -> None:
        """やり取りを終えたセッションを、次リクエストで引き当てられるよう登録する。

        キーには今回サーバーが返した assistant 応答まで含める。クライアントが応答を
        加工して保存していれば次回キーが一致せず、自動的にフル送信へ落ちる。
        履歴が食い違ったまま会話が進むことを防ぐための安全弁。
        """
        if not (SESSION_REUSE and session_id and reply):
            return
        following = list(self.turns) + [Turn("assistant", reply)]
        STORE.put(
            conversation_key(self.system_prompt, self.model, self.effort, following),
            session_id,
        )

    def fallback_to_full(self) -> None:
        """resume に失敗したときに、全履歴を送る形へ落とす。"""
        if self.resume_id:
            STATS["resume_failed"] += 1
            STORE.drop_session(self.resume_id)
        self.resume_id = None
        self.prompt = render_turns(self.turns)
        self.content = render_blocks(self.turns)


def build_plan(
    convo: Conversation, model: Optional[str], effort: Optional[str]
) -> Plan:
    """セッションを引き当てて実行プランを決める。

    引き当てと Plan.register は同一会話ロックの中で行う必要がある(呼び出し側の責務)。
    """
    turns = convo.turns
    resume_id: Optional[str] = None
    delta: Sequence[Turn] = turns

    if SESSION_REUSE:
        # 末尾から遡り、既知のプレフィックスに一致するセッションを探す。
        oldest = max(1, len(turns) - LOOKBACK_TURNS)
        for k in range(len(turns) - 1, oldest - 1, -1):
            session_id = STORE.get(
                conversation_key(convo.system_prompt, model, effort, turns[:k])
            )
            if session_id:
                resume_id = session_id
                delta = turns[k:]
                break
        STATS["hit" if resume_id else "miss"] += 1

    return Plan(
        prompt=render_turns(delta),
        system_prompt=convo.system_prompt,
        turns=list(turns),
        content=render_blocks(delta),
        model=model,
        effort=effort,
        resume_id=resume_id,
    )
