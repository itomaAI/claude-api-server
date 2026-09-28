"""Claude Code CLI (サブスクリプション課金) を OpenAI 互換 API として中継するサーバー。

    uvicorn server:app --host 127.0.0.1 --port 8811

前提と設計は README.md を参照。要点だけ:

- 認証なし・ローカル限定利用を前提とする。
- --bare は使わない: bare は ANTHROPIC_API_KEY 認証を強制し、OAuth(サブスク課金)を
  無効化するため。
- クライアントは毎回全会話履歴を送るステートレス方式のままでよい。サーバー側だけが
  セッションを持ち、履歴のハッシュで既存セッションを引き当てて差分だけを積む
  (プロンプトキャッシュを効かせるため)。詳細は conversation.py / sessions.py。
- 状態はプロセス内メモリにあるため、必ずワーカー1本で動かすこと。
"""

import asyncio
import json
import os
import re
import time
import uuid
from typing import Any, AsyncIterator, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import claude
from config import (
    AVAILABLE_MODELS,
    ERROR_DIR,
    ERROR_DUMP,
    SESSION_REUSE,
    ensure_dirs,
)
from conversation import (
    Conversation,
    InvalidConversation,
    build_plan,
    conversation_root_key,
    render_messages,
)
from sessions import STATS, STORE, conversation_lock

ensure_dirs()

app = FastAPI(title="Claude Code Pseudo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# リクエスト / レスポンス
# --------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    # OpenAI 互換クライアントは content を単純な文字列ではなく
    # [{"type": "text", "text": "..."}] 等の content-parts 配列で送ってくることがある。
    # どちらも受け付ける。
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
    キャッシュ読込 の3種に分かれて返るため、prompt_tokens はその合計
    (= 実際に消費された総入力量)。内訳は prompt_tokens_details に入れる。
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
    root = conversation_root_key(convo.system_prompt, req.model, req.effective_effort, convo.turns)
    return root, convo


# --------------------------------------------------------------------------
# 非ストリーミング
# --------------------------------------------------------------------------

async def complete(req: ChatRequest) -> claude.RunResult:
    root, convo = prepare(req)

    # 引き当て・実行・登録を会話単位で直列化する。別会話同士は並行に走る。
    async with conversation_lock(root):
        plan = build_plan(convo, req.model, req.effective_effort)

        try:
            result = await claude.run(plan)
            if result.broken and plan.resume_id:
                # セッションが消えている / セッションを再生すると API に弾かれる等。
                # 引き当てたセッションを捨てて全履歴を送り直す。
                dump_failure(plan, result)
                plan.fallback_to_full()
                result = await claude.run(plan)
        except claude.ClaudeError as exc:
            raise HTTPException(502, str(exc)) from exc

        if result.broken:
            dump_failure(plan, result)
            raise HTTPException(502, result.describe_error())

        reply, was_cut = claude.cut_text(result.reply, req.x_cut_after)
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


# --------------------------------------------------------------------------
# ストリーミング (SSE)
# --------------------------------------------------------------------------

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
        return self._frame({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]})

    def finish(self) -> str:
        return self._frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})

    def usage(self, usage: dict) -> str:
        return self._frame({"choices": [], "usage": to_openai_usage(usage)})

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"


async def stream_completion(req: ChatRequest) -> AsyncIterator[str]:
    root, convo = prepare(req)
    sse = SSEWriter(req.model or "claude-code-default")

    emitted = False
    result = claude.RunResult()

    # SSE を返し終えるまでロックを保持するので、同一会話の次リクエストは
    # 前のターンが登録され終わってから引き当てを行える。
    async with conversation_lock(root):
        plan = build_plan(convo, req.model, req.effective_effort)

        async for kind, payload in claude.stream(plan, cut_after=req.x_cut_after):
            if kind == "text":
                emitted = True
                yield sse.text(payload)
            else:
                result = payload

        # まだ1バイトも返していないなら、全履歴で安全にやり直せる。
        if plan.resume_id and result.broken and not emitted:
            dump_failure(plan, result)
            plan.fallback_to_full()
            async for kind, payload in claude.stream(plan, cut_after=req.x_cut_after):
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


# --------------------------------------------------------------------------
# エンドポイント
# --------------------------------------------------------------------------

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
            text = await claude.usage_report()
        except claude.ClaudeError as exc:
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
