#!/usr/bin/env python3
"""Claude Code CLI (サブスクリプション課金) を OpenAI 互換 API として中継するローカルサーバー。

単一ファイル版。依存は fastapi と uvicorn のみ。

    pip install fastapi uvicorn
    python3 claude_api_server.py --port 8811

    curl -s http://127.0.0.1:8811/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"sonnet","messages":[{"role":"user","content":"日本の首都は?"}]}'

前提:
  - 認証なし・ローカル限定利用。5時間レート制限は全クライアントで共有される。
  - 状態はプロセス内メモリにあるため、必ずワーカー1本で動かすこと。
  - claude CLI がインストール済みで、OAuth ログイン(サブスク)が済んでいること。

対応していないもの:
  tools / function calling(黙って無視される)、temperature や max_tokens 等の
  サンプリングパラメータ(claude -p に対応するフラグが無い)、role: "tool"、
  末尾が assistant のメッセージ列(prefill)、/v1/embeddings。

セッション再利用によるプロンプトキャッシュ最適化:
  Anthropic のプロンプトキャッシュはプレフィックス一致で、境界はコンテンツブロック単位。
  全履歴を1つのテキストとして毎回渡すと、会話が1ターン伸びるだけでブロック全体が
  別物になり、履歴全体がキャッシュ書き込み(通常入力の1.25〜2倍課金相当)として再送される。

  そこでクライアントはステートレスのまま、サーバー側だけがセッションを持つ。
  会話履歴のハッシュ -> session_id を保持し、一致すれば --resume で差分ターンのみを積む。
  実測では 1ターンあたりのキャッシュ書き込みが 13,571 -> 21 トークンになった。

  ハッシュのキーには「サーバーが返した assistant 応答」も含める。クライアントが応答を
  加工して保存していた場合はキーが一致せず、自動的にフル送信へ落ちる。履歴が食い違った
  まま会話が進むことはない。
"""

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ==========================================================================
# 設定
# ==========================================================================

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")

# claude を実行する作業ディレクトリ。空のディレクトリを使い、他プロジェクトの
# CLAUDE.md が意図せず読み込まれないようにする。
WORKDIR = os.environ.get(
    "CLAUDE_API_WORKDIR", os.path.expanduser("~/.claude-api-workdir")
)
ATTACH_DIR = os.path.join(WORKDIR, "attachments")
TMP_DIR = os.path.join(WORKDIR, "tmp")

# セッション再利用。0 で無効化し、毎回フル送信する従来動作に戻る。
SESSION_REUSE = os.environ.get("CLAUDE_API_SESSION_REUSE", "1") != "0"
MAX_SESSIONS = int(os.environ.get("CLAUDE_API_MAX_SESSIONS", "64"))
SESSION_TTL = float(os.environ.get("CLAUDE_API_SESSION_TTL", "86400"))

# 直近何ターン分さかのぼってプレフィックス一致を探すか。
LOOKBACK_TURNS = 8

# 同時に保持する会話ロックの上限。
MAX_CONV_LOCKS = 256

# stream-json の1行は Read ツール経由の base64 を含みうるので、asyncio の
# readline() 既定上限(64KB)では足りない。
STREAM_LINE_LIMIT = 100 * 1024 * 1024

# /v1/models で返すモデル。claude CLI にモデル列挙コマンドは無いため定数で持つ。
DEFAULT_MODELS = (
    "sonnet",
    "opus",
    "haiku",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-haiku-4-5",
)
AVAILABLE_MODELS = tuple(
    m.strip()
    for m in os.environ.get("CLAUDE_API_MODELS", ",".join(DEFAULT_MODELS)).split(",")
    if m.strip()
)

# 失敗したターンの再現材料を書き出す先。0 で無効化。
ERROR_DIR = os.path.join(WORKDIR, "errors")
ERROR_DUMP = os.environ.get("CLAUDE_API_ERROR_DUMP", "1") != "0"

os.makedirs(ATTACH_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

DATA_URI_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)


# ==========================================================================
# セッション対応表と会話ロック
# ==========================================================================

# セッション引き当ての実績。hit_rate が低い場合、クライアントが assistant 応答を
# 加工して保存している可能性が高い。
STATS = {"hit": 0, "miss": 0, "resume_failed": 0, "truncated": 0}


def session_file_path(session_id: str) -> str:
    """claude がセッションを永続化する .jsonl のパス。"""
    encoded = WORKDIR.replace("/", "-").replace(".", "-")
    return os.path.join(
        os.path.expanduser("~/.claude/projects"), encoded, f"{session_id}.jsonl"
    )


class SessionStore:
    """会話プレフィックスのハッシュ -> session_id。LRU + TTL。"""

    def __init__(self, max_items: int = MAX_SESSIONS, ttl: float = SESSION_TTL):
        self._items: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
        self._max = max_items
        self._ttl = ttl

    def get(self, key: str) -> Optional[str]:
        item = self._items.get(key)
        if item is None:
            return None
        session_id, stored_at = item
        if time.time() - stored_at > self._ttl:
            del self._items[key]
            self._discard_file(session_id)
            return None
        self._items.move_to_end(key)
        return session_id

    def put(self, key: str, session_id: str) -> None:
        self._items[key] = (session_id, time.time())
        self._items.move_to_end(key)
        while len(self._items) > self._max:
            _, (evicted_id, _) = self._items.popitem(last=False)
            self._discard_file_if_unreferenced(evicted_id)

    def drop_session(self, session_id: str) -> None:
        """resume に失敗したセッションを引き当て対象から外し、ファイルも消す。"""
        for key in [k for k, (sid, _) in self._items.items() if sid == session_id]:
            del self._items[key]
        self._discard_file(session_id)

    def size(self) -> int:
        return len(self._items)

    def _discard_file_if_unreferenced(self, session_id: str) -> None:
        # 同じ session_id が別のキーからも参照されている間はファイルを消さない。
        if any(sid == session_id for sid, _ in self._items.values()):
            return
        self._discard_file(session_id)

    @staticmethod
    def _discard_file(session_id: str) -> None:
        try:
            os.remove(session_file_path(session_id))
        except OSError:
            pass


STORE = SessionStore()

# 同一会話への同時リクエストを直列化するロック。「引き当て -> 実行 -> 登録」を
# まとめて保護しないと、2本が同じセッションを引き当て、後発が先発の進めた
# セッションに差分を積んで履歴が混ざる。
_CONV_LOCKS: "OrderedDict[str, asyncio.Lock]" = OrderedDict()


def conversation_lock(key: str) -> asyncio.Lock:
    lock = _CONV_LOCKS.get(key)
    if lock is None:
        lock = _CONV_LOCKS[key] = asyncio.Lock()
    _CONV_LOCKS.move_to_end(key)

    # 使用中でない古いロックだけ捨てる。保持中のロックを消すと直列化が壊れる。
    if len(_CONV_LOCKS) > MAX_CONV_LOCKS:
        for stale in [
            k for k in list(_CONV_LOCKS) if k != key and not _CONV_LOCKS[k].locked()
        ]:
            del _CONV_LOCKS[stale]
            if len(_CONV_LOCKS) <= MAX_CONV_LOCKS:
                break
    return lock


# ==========================================================================
# メッセージのレンダリングと実行プラン
# ==========================================================================

Content = Union[str, list[dict[str, Any]]]


class RawMessage(Protocol):
    """OpenAI 形式の1メッセージ。"""

    role: str
    content: Content


class InvalidConversation(ValueError):
    """クライアントの messages が受け付けられない形だった。"""


def save_data_uri(data_uri: str, hint_filename: Optional[str] = None) -> Optional[str]:
    """data: URI をデコードして ATTACH_DIR に保存し、絶対パスを返す。

    ファイル名は内容のハッシュから決める。同じ添付が常に同じパスへ落ちるので、
    会話ハッシュが安定し(= セッション再利用が効き)、保存も重複しない。
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
                texts.append(
                    f"\n[Attached image, read it with the Read tool: {path}]\n"
                )
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


# ==========================================================================
# claude CLI の起動
#
# プロンプトは CLI 引数ではなく stdin で渡す。CLI 引数には Linux の1引数あたりの
# 長さ上限(MAX_ARG_STRLEN, 通常128KB)があり、会話履歴を載せると
# "OSError: [Errno 7] Argument list too long" になるため。
# 同じ理由で system prompt は --system-prompt-file 経由にしている。
# ==========================================================================


class ClaudeError(RuntimeError):
    """claude の起動・出力の解釈に失敗した。"""


@dataclass
class RunResult:
    """1回の claude 実行の結果。"""

    returncode: int = 0
    session_id: Optional[str] = None
    usage: dict = field(default_factory=dict)
    reply: str = ""
    stderr: bytes = b""
    is_error: bool = False
    error_message: str = ""
    # クライアント指定の切り位置(x_cut_after)で応答を打ち切った。
    # このときセッションには切る前の全文が残っているため、登録せず捨てること。
    truncated: bool = False

    @property
    def failed(self) -> bool:
        """プロセスが異常終了した。"""
        return self.returncode != 0

    @property
    def broken(self) -> bool:
        """このターンは失敗した(プロセス異常終了、または claude が API エラーを報告)。

        is_error は終了コード 0 でも立つことがあるので、両方を見ないと
        壊れたセッションをそのまま登録してしまう。
        """
        return self.failed or self.is_error

    def describe_error(self) -> str:
        if self.error_message:
            return self.error_message
        if self.failed:
            return f"claude exited {self.returncode}: {self.stderr_text()}"
        return "unknown error from claude"

    def stderr_text(self, limit: int = 2000) -> str:
        return self.stderr.decode(errors="replace")[:limit]



def _first_cut_end(text: str, cuts: Optional[list[str]]) -> Optional[int]:
    """最初に現れた切り文字列の終端インデックス。無ければ None。

    同じ開始位置に複数が一致したら長いほうを採る。片方がもう片方の接頭辞のとき
    (例: "<yield" と "<yield />")、短いほうを採るとタグの途中で切ってしまうため。
    """
    best: Optional[tuple[int, int]] = None  # (start, -end): 開始が早い順、同着は長い順
    for c in cuts or []:
        if not c:
            continue
        i = text.find(c)
        if i >= 0:
            cand = (i, -(i + len(c)))
            if best is None or cand < best:
                best = cand
    return None if best is None else -best[1]


def cut_text(text: str, cuts: Optional[list[str]]) -> tuple[str, bool]:
    """cuts のいずれかが最初に現れた「直後」で切る。戻り値は (本文, 打ち切ったか)。

    OpenAI の stop と意味論が違うことに注意: stop は一致文字列を含めないが、
    こちらは一致文字列を残してその直後で切る(クライアントのパーサが
    終端タグ本体を必要とするため)。だから stop という名前を使っていない。

    「打ち切ったか」は一致の有無ではなく、切り位置の後ろに空白以外の中身が
    あったかで決まる。終端タグで正常に終わる応答は毎回一致するので、一致だけで
    打ち切り扱いにするとセッションが毎ターン破棄されてしまう。後ろが改行だけ、
    というのもごく普通に起きるため strip で判定する。
    """
    end = _first_cut_end(text, cuts)
    if end is None:
        return text, False
    head, tail = text[:end], text[end:]
    return head, tail.strip() != ""


class CutFilter:
    """ストリーミング応答を逐次監視し、指定文字列の直後で流すのをやめる。

    一致はチャンクをまたいで起こりうるため、最長の指定文字列 - 1 文字ぶんを
    手元に留めてから流す。切り位置に達する(matched)と以降は何も流さないが、
    捨てたテキストは discarded に貯める。打ち切り(truncating)が確定するのは
    捨てた中に空白以外が現れたときだけ。応答が切り文字列で正常に終わる場合は
    matched のまま truncating にならず、通常どおり result まで読み切れる。
    """

    def __init__(self, cuts: Optional[list[str]]):
        self.cuts = [c for c in (cuts or []) if c]
        self.keep = max((len(c) for c in self.cuts), default=1) - 1
        self.pending = ""
        self.matched = False
        self.discarded = ""

    @property
    def truncating(self) -> bool:
        """本物の打ち切りが確定した(切り位置の後ろに空白以外の中身が来た)。"""
        return self.matched and self.discarded.strip() != ""

    def feed(self, text: str) -> str:
        """新しいテキストを受け取り、流してよいぶんを返す。"""
        if not self.cuts:
            return text
        if self.matched:
            self.discarded += text
            return ""
        self.pending += text
        end = _first_cut_end(self.pending, self.cuts)
        if end is not None:
            self.matched = True
            out, self.discarded = self.pending[:end], self.pending[end:]
            self.pending = ""
            return out
        if self.keep and len(self.pending) > self.keep:
            out, self.pending = self.pending[: -self.keep], self.pending[-self.keep :]
            return out
        if not self.keep:
            out, self.pending = self.pending, ""
            return out
        return ""

    def flush(self) -> str:
        """ストリームが切り位置に達さず終わったとき、留めていた残りを返す。"""
        out, self.pending = self.pending, ""
        return out


@contextmanager
def _system_prompt_file(system_prompt: str) -> Iterator[Optional[str]]:
    if not system_prompt:
        yield None
        return
    fd, path = tempfile.mkstemp(dir=TMP_DIR, suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(system_prompt)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def build_command(plan: Plan, system_prompt_file: Optional[str]) -> list[str]:
    """claude の起動引数を組み立てる(プロンプト本体は stdin なのでここには入れない)。"""
    cmd = [CLAUDE_BIN, "-p"]

    if plan.resume_id:
        cmd += ["--resume", plan.resume_id]
    elif not SESSION_REUSE:
        # セッションを残さない従来動作。再利用するにはセッション永続化が要る。
        cmd += ["--no-session-persistence"]

    if plan.has_attachment:
        # 添付があるときだけ Read を許可し、非対話でも確認待ちにならないようにする。
        cmd += ["--tools", "Read", "--permission-mode", "dontAsk"]
    else:
        cmd += ["--tools", ""]

    if system_prompt_file:
        cmd += ["--system-prompt-file", system_prompt_file]
    if plan.model:
        cmd += ["--model", plan.model]
    if plan.effort:
        cmd += ["--effort", plan.effort]
    return cmd


async def _spawn(cmd: list[str], limit: Optional[int] = None):
    kwargs = {"limit": limit} if limit else {}
    return await asyncio.create_subprocess_exec(
        *cmd,
        cwd=WORKDIR,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )


async def usage_report() -> str:
    """/usage スラッシュコマンドを headless で実行し、レポート本文を返す。

    /usage はローカルコマンドとして処理されるため、モデル呼び出しは発生せず
    トークンも消費しない。-p モードで動くことはドキュメント化されていない挙動
    なので、CLI のバージョンアップで動かなくなる可能性がある(2.1.237 で確認)。
    """
    cmd = [CLAUDE_BIN, "-p", "--output-format", "json",
           "--tools", "", "--no-session-persistence"]
    proc = await _spawn(cmd)
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=b"/usage"), timeout=60
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ClaudeError("claude /usage timed out")
    if proc.returncode != 0:
        raise ClaudeError(
            f"claude exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}"
        )
    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError as exc:
        raise ClaudeError("could not parse claude /usage output") from exc
    if data.get("is_error"):
        raise ClaudeError(str(data.get("result"))[:500])
    return data.get("result") or ""


async def claude_run(plan: Plan) -> RunResult:
    """--output-format json で1回実行し、結果をまとめて返す。"""
    with _system_prompt_file(plan.system_prompt) as spf:
        cmd = build_command(plan, spf) + ["--output-format", "json"]
        proc = await _spawn(cmd)
        stdout, stderr = await proc.communicate(input=plan.prompt.encode())

    result = RunResult(returncode=proc.returncode, stderr=stderr)
    if result.failed:
        return result

    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError as exc:
        raise ClaudeError(
            f"could not parse claude output: {stdout.decode(errors='replace')[:2000]}"
        ) from exc

    result.session_id = data.get("session_id")
    result.usage = data.get("usage") or {}
    result.is_error = bool(data.get("is_error"))
    if result.is_error:
        result.error_message = str(data.get("result") or "unknown error")
    else:
        result.reply = data.get("result", "")
    return result


async def claude_stream(
    plan: Plan, cut_after: Optional[list[str]] = None
) -> AsyncIterator[tuple[str, object]]:
    """--output-format stream-json で実行し、テキスト差分を逐次流す。

    ("text", str) を出力があるたびに、最後に必ず ("result", RunResult) を1回 yield する。
    cut_after が指定されていれば CutFilter を通す。切り位置に達しても、応答が
    そこで終わるだけなら通常どおり result まで読み切る(usage も取れる)。
    切り位置の後ろに空白以外の中身が来て打ち切りが確定したときだけ
    サブプロセスを終了する(result.truncated が立ち、このときは usage を失う)。
    """
    result = RunResult()
    reply_parts: list[str] = []
    cutter = CutFilter(cut_after)

    with _system_prompt_file(plan.system_prompt) as spf:
        cmd = build_command(plan, spf) + [
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
        ]
        proc = await _spawn(cmd, limit=STREAM_LINE_LIMIT)

        proc.stdin.write(plan.prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()

        stderr_chunks: list[bytes] = []

        async def drain_stderr() -> None:
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)

        stderr_task = asyncio.create_task(drain_stderr())
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # session_id は最初の system/init イベントから取れる。result まで
                # 到達しない打ち切り時でも、汚れたセッションを特定して消すために要る。
                if not result.session_id and event.get("session_id"):
                    result.session_id = event.get("session_id")

                if event.get("type") == "stream_event":
                    inner = event.get("event", {})
                    if inner.get("type") == "content_block_delta":
                        delta = inner.get("delta", {})
                        if delta.get("type") == "text_delta":
                            out = cutter.feed(delta.get("text", ""))
                            if out:
                                reply_parts.append(out)
                                yield "text", out
                            if cutter.truncating:
                                # 切り位置の後ろに中身が来た = 本物の打ち切りが確定。
                                # 以降の生成は無駄なので止める。切り文字列で正常に
                                # 終わるだけの応答はここに来ず、result まで読み切って
                                # usage を取る。
                                result.truncated = True
                                proc.terminate()
                                break
                elif event.get("type") == "result":
                    result.session_id = event.get("session_id")
                    result.usage = event.get("usage") or {}
                    result.is_error = bool(event.get("is_error"))
                    if result.is_error:
                        # ここで text として流すと emitted 扱いになり、
                        # 呼び出し側がフルリトライできなくなる。結果に持たせるだけにする。
                        result.error_message = str(event.get("result") or "unknown error")
        finally:
            # terminate が SIGTERM を無視されると永久に待つので上限を付ける。
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            await stderr_task

    if not result.truncated:
        # 切り位置に達さないまま終わったら、CutFilter が留めていた末尾を流す。
        tail = cutter.flush()
        if tail:
            reply_parts.append(tail)
            yield "text", tail

    # 打ち切りはこちら都合の terminate なので、シグナル死を失敗扱いにしない。
    result.returncode = 0 if result.truncated else proc.returncode
    result.stderr = b"".join(stderr_chunks)
    result.reply = "".join(reply_parts)
    yield "result", result


# ==========================================================================
# HTTP レイヤ
# ==========================================================================

app = FastAPI(title="Claude Code Pseudo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    # OpenAI 互換クライアントは content を単純な文字列ではなく
    # [{"type": "text", "text": "..."}] 等の content-parts 配列で送ってくることがある。
    content: Union[str, list[dict[str, Any]]]


class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = None
    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None
    # OpenAI 互換名。標準クライアントは reasoning_effort で送ってくる。
    # "minimal" は claude --effort に無いので low に読み替える。
    # effort と両方指定されたときはサーバー固有の effort を優先する。
    reasoning_effort: Optional[
        Literal["minimal", "low", "medium", "high", "xhigh", "max"]
    ] = None

    @property
    def effective_effort(self) -> Optional[str]:
        if self.effort:
            return self.effort
        if self.reasoning_effort:
            return "low" if self.reasoning_effort == "minimal" else self.reasoning_effort
        return None
    stream: Optional[bool] = False
    stream_options: Optional[dict] = None
    # 応答の切り位置。いずれかの文字列が最初に現れた「直後」で応答を切り、
    # そのターンのセッションは登録せず破棄する(セッション側には切る前の全文が
    # 残ってしまうため。残すと次ターンの --resume で全文が文脈へ戻る)。
    # OpenAI の stop とは意味論が違う: stop は一致文字列を含めないが、これは
    # 一致文字列を残す。だから別名にしてある。
    x_cut_after: Optional[list[str]] = None
    # temperature / max_tokens / tools などの OpenAI 標準パラメータは
    # claude -p に対応するものが無いため、受け取っても無視する。


def to_openai_usage(usage: dict) -> dict:
    """claude の usage を OpenAI 形式に変換する。

    Anthropic のキャッシュ機構により入力トークンは 新規 / キャッシュ書込 /
    キャッシュ読込 の3種に分かれて返るため、prompt_tokens はその合計。
    """
    fresh = usage.get("input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    prompt_tokens = fresh + cache_write + cache_read
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
        },
    }


def dump_failure(plan, result) -> Optional[str]:
    """失敗したターンの再現材料を WORKDIR/errors/ に書き出す。

    claude が API エラーを報告したとき、実際に何を送ったのかが分からないと
    原因を追えないため。CLAUDE_API_ERROR_DUMP=0 で無効化できる。
    会話本文をそのまま含むので、共有する前に中身を確認すること。
    """
    if not ERROR_DUMP:
        return None
    try:
        os.makedirs(ERROR_DIR, exist_ok=True)
        path = os.path.join(ERROR_DIR, f"{int(time.time() * 1000)}.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "error": result.describe_error(),
                    "returncode": result.returncode,
                    "is_error": result.is_error,
                    "stderr": result.stderr_text(),
                    "resumed_session": plan.resume_id,
                    "model": plan.model,
                    "effort": plan.effort,
                    "has_attachment": plan.has_attachment,
                    "prompt_sent": plan.prompt,
                    "system_prompt": plan.system_prompt,
                    "history": [
                        {"role": t.role, "chars": len(t.text), "text": t.text}
                        for t in plan.turns
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return path
    except OSError:
        return None


def prepare(req: ChatRequest) -> tuple[str, Conversation]:
    """messages を検証・レンダリングし、(会話ロックのキー, Conversation) を返す。"""
    try:
        convo = render_messages(req.messages)
    except InvalidConversation as exc:
        raise HTTPException(400, str(exc)) from exc
    root = conversation_root_key(
        convo.system_prompt, req.model, req.effective_effort, convo.turns
    )
    return root, convo


async def complete(req: ChatRequest) -> RunResult:
    root, convo = prepare(req)

    # 引き当て・実行・登録を会話単位で直列化する。別会話同士は並行に走る。
    async with conversation_lock(root):
        plan = build_plan(convo, req.model, req.effective_effort)

        try:
            result = await claude_run(plan)
            if result.broken and plan.resume_id:
                # セッションが消えている / セッションを再生すると API に弾かれる等。
                # 引き当てたセッションを捨てて全履歴を送り直す。
                dump_failure(plan, result)
                plan.fallback_to_full()
                result = await claude_run(plan)
        except ClaudeError as exc:
            raise HTTPException(502, str(exc)) from exc

        if result.broken:
            dump_failure(plan, result)
            raise HTTPException(502, result.describe_error())

        reply, was_cut = cut_text(result.reply, req.x_cut_after)
        # 一致があれば本文は常に切り位置まで(後ろが空白だけでも落とす)。
        # クライアントが保存するのも切り位置までの本文なので、登録キーが
        # クライアントの再送と一致しやすくなる。
        result.reply = reply
        if was_cut:
            # resume は成功したが応答が切り位置を越えた。セッションには切る前の
            # 全文が assistant ターンとして残っており、次の --resume でそれが
            # モデルの文脈へ戻ってしまうため、登録せず破棄する。
            # (resume 失敗時の fallback_to_full とは別系統の分岐)
            result.truncated = True
            STATS["truncated"] += 1
            if result.session_id:
                STORE.drop_session(result.session_id)
        else:
            # 失敗したターンは登録しない。登録すると壊れたセッションを
            # 次ターン以降も掴み続け、同じエラーが出続ける。
            plan.register(result.session_id, result.reply)

    return result


class SSEWriter:
    """OpenAI の chat.completion.chunk 形式で SSE 行を作る。"""

    def __init__(self, model: str):
        self.id = f"chatcmpl-{uuid.uuid4()}"
        self.created = int(time.time())
        self.model = model
        self._role_sent = False

    def _frame(self, body: dict) -> str:
        payload = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            **body,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def text(self, content: str) -> str:
        delta = {"content": content}
        if not self._role_sent:
            self._role_sent = True
            delta = {"role": "assistant", **delta}
        return self._frame(
            {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        )

    def finish(self) -> str:
        return self._frame(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        )

    def usage(self, usage: dict) -> str:
        return self._frame({"choices": [], "usage": to_openai_usage(usage)})

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"


async def stream_completion(req: ChatRequest) -> AsyncIterator[str]:
    root, convo = prepare(req)
    sse = SSEWriter(req.model or "claude-code-default")

    emitted = False
    result = RunResult()

    # SSE を返し終えるまでロックを保持するので、同一会話の次リクエストは
    # 前のターンが登録され終わってから引き当てを行える。
    async with conversation_lock(root):
        plan = build_plan(convo, req.model, req.effective_effort)

        async for kind, payload in claude_stream(plan, cut_after=req.x_cut_after):
            if kind == "text":
                emitted = True
                yield sse.text(payload)
            else:
                result = payload

        # まだ1バイトも返していないなら、全履歴で安全にやり直せる。
        if plan.resume_id and result.broken and not emitted:
            dump_failure(plan, result)
            plan.fallback_to_full()
            async for kind, payload in claude_stream(plan, cut_after=req.x_cut_after):
                if kind == "text":
                    emitted = True
                    yield sse.text(payload)
                else:
                    result = payload

        if result.truncated:
            # 応答が切り位置を越えた。セッションには切る前の全文が残っているため
            # 登録せず破棄する(complete() 側の分岐と同じ理由)。
            STATS["truncated"] += 1
            if result.session_id:
                STORE.drop_session(result.session_id)
        elif result.broken:
            # 失敗したターンは登録しない。登録すると壊れたセッションを
            # 次ターン以降も掴み続け、同じエラーが出続ける。
            dump_failure(plan, result)
        else:
            plan.register(result.session_id, result.reply)

    if result.broken and not emitted:
        yield sse.text(f"[claude error: {result.describe_error()}]")

    yield sse.finish()
    if req.stream_options and req.stream_options.get("include_usage"):
        yield sse.usage(result.usage)
    yield sse.done()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if req.stream:
        return StreamingResponse(stream_completion(req), media_type="text/event-stream")

    result = await complete(req)
    return {
        "id": f"chatcmpl-{result.session_id or uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "claude-code-default",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.reply},
                "finish_reason": "stop",
            }
        ],
        "usage": to_openai_usage(result.usage),
    }


@app.get("/v1/models")
async def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "anthropic"}
            for name in AVAILABLE_MODELS
        ],
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """シンプル形式(OpenAI 互換より前から使っている素の入出力)。"""
    result = await complete(req)
    return {
        "content": result.reply,
        "session_id": result.session_id,
        "usage": to_openai_usage(result.usage),
    }



# 例: "Current session: 22% used · resets Sep 8, 2:09pm (Asia/Tokyo)"
#     "Current week (all models): 36% used · resets Sep 13, 4:59pm (Asia/Tokyo)"
#     "Current week (Fable): 66% used · resets Sep 13, 4:59pm (Asia/Tokyo)"
USAGE_LINE_RE = re.compile(
    r"^Current (session|week)(?: \(([^)]+)\))?: (\d+)% used(?: · resets (.+))?$"
)


def parse_usage(text: str) -> list[dict]:
    """/usage のレポート本文から制限の消費行を抜き出す。

    表記は CLI の都合でいつでも変わりうるので、読めた行だけ返す。
    読めなくてもエンドポイントは raw で全文を返すため、クライアントは
    そちらへフォールバックできる。
    """
    limits = []
    for line in text.splitlines():
        m = USAGE_LINE_RE.match(line.strip())
        if m:
            limits.append(
                {
                    "window": m.group(1),
                    "model": m.group(2),  # week のみ。"all models" / モデル名 / None
                    "used_percent": int(m.group(3)),
                    "resets": m.group(4),
                }
            )
    return limits


# /usage は claude の起動に数秒かかるので、直近の結果を短時間キャッシュする。
_USAGE_CACHE: dict = {"at": 0.0, "payload": None}
_USAGE_LOCK = asyncio.Lock()
USAGE_CACHE_TTL = float(os.environ.get("CLAUDE_API_USAGE_CACHE_TTL", "60"))


@app.get("/usage")
async def usage(refresh: bool = False):
    """サブスクリプションの使用状況(セッション・週次制限の消費率)。

    claude CLI の /usage を headless で叩いた結果。モデル呼び出しは発生しない。
    ?refresh=1 でキャッシュを無視して取り直す。
    """
    async with _USAGE_LOCK:
        now = time.time()
        cached = _USAGE_CACHE["payload"]
        if not refresh and cached and now - _USAGE_CACHE["at"] < USAGE_CACHE_TTL:
            return {**cached, "cached": True}
        try:
            text = await usage_report()
        except ClaudeError as exc:
            raise HTTPException(502, str(exc)) from exc
        payload = {
            "limits": parse_usage(text),
            "raw": text,
            "fetched_at": int(now),
        }
        _USAGE_CACHE.update(at=now, payload=payload)
        return {**payload, "cached": False}

@app.get("/health")
async def health():
    total = STATS["hit"] + STATS["miss"]
    return {
        "status": "ok",
        "session_reuse": SESSION_REUSE,
        "sessions": STORE.size(),
        "stats": {
            **STATS,
            "hit_rate": round(STATS["hit"] / total, 3) if total else None,
        },
    }


# ==========================================================================
# エントリポイント
# ==========================================================================


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Claude Code CLI を OpenAI 互換 API として中継する"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8811)
    args = parser.parse_args()

    # 状態はプロセス内メモリにあるため、ワーカーは1本に固定する。
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

