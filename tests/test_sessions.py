"""SessionStore の LRU / TTL / ファイル後始末のテスト。"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sessions as ss  # noqa: E402
from sessions import SessionStore, conversation_lock  # noqa: E402


@pytest.fixture
def no_file_removal(monkeypatch):
    """実際のセッションファイルには触らせない。消そうとした id を記録する。"""
    removed: list[str] = []
    monkeypatch.setattr(SessionStore, "_discard_file", staticmethod(removed.append))
    return removed


def test_roundtrip(no_file_removal):
    store = SessionStore(max_items=4, ttl=3600)
    store.put("k", "sess-1")
    assert store.get("k") == "sess-1"
    assert store.get("missing") is None


def test_expired_entry_is_dropped_with_its_file(no_file_removal):
    store = SessionStore(max_items=4, ttl=-1)  # 即座に期限切れ
    store.put("k", "sess-1")
    assert store.get("k") is None
    assert no_file_removal == ["sess-1"]


def test_lru_eviction_removes_oldest(no_file_removal):
    store = SessionStore(max_items=2, ttl=3600)
    store.put("a", "sess-a")
    store.put("b", "sess-b")
    store.get("a")          # a を新しくする
    store.put("c", "sess-c")  # b が押し出される

    assert store.get("a") == "sess-a"
    assert store.get("b") is None
    assert store.get("c") == "sess-c"
    assert no_file_removal == ["sess-b"]


def test_eviction_keeps_file_while_another_key_references_it(no_file_removal):
    """同じセッションを指すキーが残っている間はファイルを消さない。"""
    store = SessionStore(max_items=2, ttl=3600)
    store.put("turn1", "sess-x")
    store.put("turn2", "sess-x")
    store.put("other", "sess-y")  # turn1 が押し出されるが sess-x はまだ turn2 が参照

    assert no_file_removal == []
    assert store.get("turn2") == "sess-x"


def test_drop_session_removes_every_key_for_it(no_file_removal):
    store = SessionStore(max_items=8, ttl=3600)
    store.put("turn1", "sess-x")
    store.put("turn2", "sess-x")
    store.put("keep", "sess-y")

    store.drop_session("sess-x")

    assert store.get("turn1") is None
    assert store.get("turn2") is None
    assert store.get("keep") == "sess-y"
    assert no_file_removal == ["sess-x"]


def test_conversation_lock_is_identical_per_key():
    assert conversation_lock("same") is conversation_lock("same")
    assert conversation_lock("a") is not conversation_lock("b")


def test_conversation_lock_registry_does_not_evict_held_locks(monkeypatch):
    monkeypatch.setattr(ss, "MAX_CONV_LOCKS", 2)
    monkeypatch.setattr(ss, "_CONV_LOCKS", ss.OrderedDict())

    async def scenario():
        held = conversation_lock("held")
        async with held:
            # 上限を超えるロックを作っても、保持中のものは捨てられない
            for i in range(10):
                conversation_lock(f"other-{i}")
            assert conversation_lock("held") is held

    asyncio.run(scenario())
