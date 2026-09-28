"""/usage レポートのパース。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server as mod_server  # noqa: E402
import claude_api_server as single  # noqa: E402

SAMPLE = """You are currently using your subscription to power your Claude Code usage

Current session: 22% used · resets Sep 8, 2:09pm (Asia/Tokyo)
Current week (all models): 36% used · resets Sep 13, 4:59pm (Asia/Tokyo)
Current week (Fable): 66% used · resets Sep 13, 4:59pm (Asia/Tokyo)

What's contributing to your limits usage?
Last 24h · 234 requests · 5 sessions
"""


@pytest.mark.parametrize("parse", [mod_server.parse_usage, single.parse_usage])
class TestParseUsage:
    def test_sample(self, parse):
        limits = parse(SAMPLE)
        assert limits == [
            {"window": "session", "model": None, "used_percent": 22,
             "resets": "Sep 8, 2:09pm (Asia/Tokyo)"},
            {"window": "week", "model": "all models", "used_percent": 36,
             "resets": "Sep 13, 4:59pm (Asia/Tokyo)"},
            {"window": "week", "model": "Fable", "used_percent": 66,
             "resets": "Sep 13, 4:59pm (Asia/Tokyo)"},
        ]

    def test_missing_resets(self, parse):
        limits = parse("Current session: 5% used")
        assert limits == [
            {"window": "session", "model": None, "used_percent": 5, "resets": None}
        ]

    def test_unrecognized_text_gives_empty(self, parse):
        # 表記が変わってパースできなくても壊れない(raw へのフォールバック前提)
        assert parse("совершенно другой формат\nusage: много") == []
        assert parse("") == []
