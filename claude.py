"""claude CLI の起動と出力の解釈。

claude は道具も MCP サーバーも持たない、純粋な LLM として起動する。
入力は stream-json(1行の JSON)。画像・文書をコンテンツブロックのまま渡すため。

プロンプトは CLI 引数ではなく stdin で渡す。CLI 引数には Linux の1引数あたりの
長さ上限(MAX_ARG_STRLEN, 通常128KB)があり、会話履歴を載せると
"OSError: [Errno 7] Argument list too long" になるため。
同じ理由で system prompt は --system-prompt-file 経由にしている。
"""

import asyncio
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional

from config import (
    CLAUDE_BIN,
    SESSION_REUSE,
    STREAM_LINE_LIMIT,
    TMP_DIR,
    WORKDIR,
)
from conversation import Plan


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
    """claude の起動引数を組み立てる(中身は stdin で渡すのでここには入れない)。

    claude は stream-json の入力に stream-json の出力を要求するので、
    出力の形もここで決める(非ストリームの経路も同じ出力から結果を拾う)。
    """
    cmd = [CLAUDE_BIN, "-p"]

    if plan.resume_id:
        cmd += ["--resume", plan.resume_id]
    elif not SESSION_REUSE:
        # セッションを残さない従来動作。再利用するにはセッション永続化が要るので付けない。
        cmd += ["--no-session-persistence"]

    # 純粋な LLM API として動かす: 組み込みの道具も MCP サーバーも載せない。
    # --tools "" が外すのは組み込みの道具だけで、利用者の設定や claude.ai 側で
    # 繋いだ MCP サーバーは残る。--mcp-config を渡さずに --strict-mcp-config を
    # 付けると MCP は 0 個になる(2.1.280 で確認)。
    cmd += ["--tools", "", "--strict-mcp-config"]

    cmd += [
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
    ]

    if system_prompt_file:
        cmd += ["--system-prompt-file", system_prompt_file]
    if plan.model:
        cmd += ["--model", plan.model]
    if plan.effort:
        cmd += ["--effort", plan.effort]
    return cmd


def stdin_payload(plan: Plan) -> bytes:
    """claude の stdin へ渡す stream-json の1行(user メッセージ1つ)。"""
    message = {
        "type": "user",
        "message": {"role": "user", "content": plan.blocks},
    }
    return (json.dumps(message, ensure_ascii=False) + "\n").encode()


def parse_result_event(stdout: bytes) -> Optional[dict]:
    """stream-json の出力から最後の result イベントを拾う。無ければ None。"""
    found: Optional[dict] = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            found = event
    return found


async def _spawn(cmd: list[str], limit: Optional[int] = None) -> asyncio.subprocess.Process:
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
           "--tools", "", "--strict-mcp-config", "--no-session-persistence"]
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


async def run(plan: Plan) -> RunResult:
    """1回実行し、結果をまとめて返す(出力は stream-json。result イベントを拾う)。"""
    with _system_prompt_file(plan.system_prompt) as spf:
        cmd = build_command(plan, spf)
        proc = await _spawn(cmd, limit=STREAM_LINE_LIMIT)
        stdout, stderr = await proc.communicate(input=stdin_payload(plan))

    result = RunResult(returncode=proc.returncode, stderr=stderr)
    if result.failed:
        return result

    data = parse_result_event(stdout)
    if data is None:
        raise ClaudeError(
            f"could not parse claude output: {stdout.decode(errors='replace')[:2000]}"
        )

    result.session_id = data.get("session_id")
    result.usage = data.get("usage") or {}
    result.is_error = bool(data.get("is_error"))
    if result.is_error:
        result.error_message = str(data.get("result") or "unknown error")
    else:
        result.reply = data.get("result") or ""
    return result


async def stream(
    plan: Plan, cut_after: Optional[list[str]] = None
) -> AsyncIterator[tuple[str, object]]:
    """部分メッセージつきの stream-json で実行し、テキスト差分を逐次流す。

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
        cmd = build_command(plan, spf) + ["--include-partial-messages"]
        proc = await _spawn(cmd, limit=STREAM_LINE_LIMIT)

        proc.stdin.write(stdin_payload(plan))
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
