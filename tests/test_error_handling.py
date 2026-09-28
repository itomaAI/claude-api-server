"""失敗したターンの扱い: 壊れたセッションを登録し続けないこと。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import claude as mod_claude  # noqa: E402
import claude_api_server as single  # noqa: E402


@pytest.mark.parametrize("RunResult", [mod_claude.RunResult, single.RunResult])
def test_broken_covers_both_failure_modes(RunResult):
    assert RunResult().broken is False
    # プロセス異常終了
    assert RunResult(returncode=1).broken is True
    # 終了コード 0 でも claude が API エラーを報告することがある
    assert RunResult(returncode=0, is_error=True).broken is True


@pytest.mark.parametrize("RunResult", [mod_claude.RunResult, single.RunResult])
def test_describe_error_prefers_api_message(RunResult):
    r = RunResult(returncode=0, is_error=True,
                  error_message="API Error: 400 messages: text content blocks must be non-empty")
    assert "400" in r.describe_error()

    r2 = RunResult(returncode=1, stderr=b"boom")
    assert "exited 1" in r2.describe_error() and "boom" in r2.describe_error()

    assert RunResult().describe_error() == "unknown error from claude"


@pytest.mark.parametrize("RunResult", [mod_claude.RunResult, single.RunResult])
def test_api_error_does_not_leak_into_reply(RunResult):
    """エラー文を reply に混ぜるとセッション登録キーに紛れ込む。"""
    r = RunResult(returncode=0, is_error=True, error_message="boom")
    assert r.reply == ""
