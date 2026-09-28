"""OpenAI 形式のメッセージ列を claude へ渡す形に変換し、
セッションを引き当てて実行プランを決めるところまで。
"""

import base64
import hashlib
import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, Union

from config import ATTACH_DIR, LOOKBACK_TURNS, SESSION_REUSE
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
# 添付ファイル
# --------------------------------------------------------------------------

def save_data_uri(data_uri: str, hint_filename: Optional[str] = None) -> Optional[str]:
    """data: URI をデコードして ATTACH_DIR に保存し、絶対パスを返す。

    ファイル名は内容のハッシュから決める。同じ添付が常に同じパスへ落ちるので、
    会話ハッシュが安定し(= セッション再利用が効き)、保存も重複しない。
    base64 の data URI でなければ None。
    """
    m = DATA_URI_RE.match(data_uri)
    if not m or not m.group(2):
        return None

    mime = m.group(1) or "application/octet-stream"
    try:
        raw = base64.b64decode(m.group(3))
    except Exception:
        return None

    ext = os.path.splitext(hint_filename)[1] if hint_filename else ""
    if not ext:
        ext = mimetypes.guess_extension(mime.split(";")[0].strip()) or ".bin"

    path = os.path.join(ATTACH_DIR, f"{hashlib.sha256(raw).hexdigest()[:32]}{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(raw)
    return path


# --------------------------------------------------------------------------
# メッセージ -> ターン
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    text: str
    has_attachment: bool = False


def render_content(content: Content) -> tuple[str, bool]:
    """content を claude へ渡すテキストにする。戻り値は (text, 添付を含むか)。

    画像・ファイルは data URI をディスクへ落とし、Read ツールで読ませるための
    案内文に置き換える(claude -p にネイティブの添付入力が無いため)。
    """
    if isinstance(content, str):
        return content, False

    texts: list[str] = []
    has_attachment = False
    for part in content:
        ptype = part.get("type")
        if ptype == "text":
            texts.append(part.get("text", ""))
        elif ptype == "image_url":
            path = save_data_uri((part.get("image_url") or {}).get("url", ""))
            if path:
                has_attachment = True
                texts.append(f"\n[Attached image, read it with the Read tool: {path}]\n")
            else:
                texts.append("[image attachment omitted: not a data URI]")
        elif ptype == "file":
            file_obj = part.get("file") or {}
            filename = file_obj.get("filename", "file")
            path = save_data_uri(file_obj.get("file_data", ""), hint_filename=filename)
            if path:
                has_attachment = True
                texts.append(
                    f"\n[Attached file '{filename}', read it with the Read tool: {path}]\n"
                )
            else:
                texts.append(f"[file attachment omitted: {filename} is not a data URI]")
    return "".join(texts), has_attachment


@dataclass(frozen=True)
class Conversation:
    turns: list[Turn]
    system_prompt: str
    system_has_attachment: bool


def render_messages(messages: Sequence[RawMessage]) -> Conversation:
    system_texts: list[str] = []
    system_attachment = False
    turns: list[Turn] = []

    for m in messages:
        text, attached = render_content(m.content)
        if m.role == "system":
            system_texts.append(text)
            system_attachment = system_attachment or attached
        else:
            turns.append(Turn(m.role, text, attached))

    if not turns or turns[-1].role != "user":
        raise InvalidConversation("messages must end with a user message")

    return Conversation(turns, "\n\n".join(system_texts), system_attachment)


def render_turns(turns: Sequence[Turn]) -> str:
    """ターン列を claude に渡す1つのプロンプト文字列にする。"""
    if len(turns) == 1 and turns[0].role == "user":
        return turns[0].text
    return "\n\n".join(
        f"{'User' if t.role == 'user' else 'Assistant'}: {t.text}" for t in turns
    )


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

    prompt: str
    system_prompt: str
    has_attachment: bool
    turns: list[Turn]
    model: Optional[str] = None
    effort: Optional[str] = None
    resume_id: Optional[str] = None

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
        self.has_attachment = self.has_attachment or any(
            t.has_attachment for t in self.turns
        )


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
        has_attachment=convo.system_has_attachment
        or any(t.has_attachment for t in delta),
        turns=list(turns),
        model=model,
        effort=effort,
        resume_id=resume_id,
    )
