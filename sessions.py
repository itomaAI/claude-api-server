"""セッション対応表と、会話単位の直列化ロック。

claude のセッションは ~/.claude/projects/<cwd由来の名前>/<session_id>.jsonl に
永続化される。ここではその session_id を「会話履歴のハッシュ」から引けるように保持し、
LRU + TTL で古いものを(ファイルごと)捨てる。
"""

import asyncio
import os
import time
from collections import OrderedDict
from typing import Optional

from config import MAX_CONV_LOCKS, MAX_SESSIONS, SESSION_TTL, WORKDIR

# セッション引き当ての実績。hit_rate が低い場合、クライアントが assistant 応答を
# 加工して保存している(応答の切り詰めなど)可能性が高い。
STATS = {"hit": 0, "miss": 0, "resume_failed": 0, "truncated": 0}


def session_file_path(session_id: str) -> str:
    """claude がセッションを永続化する .jsonl のパス。

    ~/.claude/projects/<cwd の '/' と '.' を '-' に置換したもの>/<session_id>.jsonl
    """
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


# 同一会話への同時リクエストを直列化するロック。「引き当て→実行→登録」を
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
            k
            for k in list(_CONV_LOCKS)
            if k != key and not _CONV_LOCKS[k].locked()
        ]:
            del _CONV_LOCKS[stale]
            if len(_CONV_LOCKS) <= MAX_CONV_LOCKS:
                break
    return lock


def lock_count() -> int:
    return len(_CONV_LOCKS)
