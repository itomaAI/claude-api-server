"""x_cut_after: 応答の切り詰め。

- 一致文字列は残し、その直後で切る(OpenAI の stop と違う)
- 「打ち切り」扱いになるのは切り位置の後ろに空白以外の中身があったときだけ。
  終端タグで正常に終わる応答(LPML の全ターン)は一致しても打ち切りではない。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import claude as mod_claude  # noqa: E402
import claude_api_server as single  # noqa: E402

IMPLS = [(mod_claude.cut_text, mod_claude.CutFilter), (single.cut_text, single.CutFilter)]


@pytest.mark.parametrize("cut_text,CutFilter", IMPLS)
class TestCutText:
    def test_content_after_match_is_truncation(self, cut_text, CutFilter):
        text, was_cut = cut_text("abc<END>garbage", ["<END>"])
        assert was_cut and text == "abc<END>"

    def test_match_at_end_is_not_truncation(self, cut_text, CutFilter):
        # LPML の正常なターン: 終端タグで終わる。打ち切りではない。
        text, was_cut = cut_text("abc<END>", ["<END>"])
        assert not was_cut and text == "abc<END>"

    def test_whitespace_after_match_is_not_truncation(self, cut_text, CutFilter):
        # 終端タグの後ろに改行だけ付くのは普通に起きる
        text, was_cut = cut_text("abc<END>\n  \n", ["<END>"])
        assert not was_cut and text == "abc<END>"

    def test_no_match(self, cut_text, CutFilter):
        text, was_cut = cut_text("abc", ["<END>"])
        assert not was_cut and text == "abc"

    def test_earliest_match_wins(self, cut_text, CutFilter):
        text, was_cut = cut_text("a<B>x<A>y", ["<A>", "<B>"])
        assert was_cut and text == "a<B>"

    def test_same_start_longest_wins(self, cut_text, CutFilter):
        # 片方が接頭辞のとき、短い方を採るとタグの途中で切ってしまう
        text, was_cut = cut_text("x<yield />y", ["<yield", "<yield />"])
        assert was_cut and text == "x<yield />"

    def test_empty_and_none(self, cut_text, CutFilter):
        assert cut_text("abc", None) == ("abc", False)
        assert cut_text("abc", []) == ("abc", False)
        assert cut_text("abc", [""]) == ("abc", False)


@pytest.mark.parametrize("cut_text,CutFilter", IMPLS)
class TestCutFilter:
    @staticmethod
    def run(CutFilter, chunks, cuts):
        """チャンクを順に流し、(クライアントへ出た本文, 打ち切りになったか) を返す。

        truncating が立った時点で打ち切る(stream() 側の挙動と同じ)。
        """
        f = CutFilter(cuts)
        out = []
        for c in chunks:
            out.append(f.feed(c))
            if f.truncating:
                break
        if not f.truncating:
            out.append(f.flush())
        return "".join(out), f.truncating

    def test_content_after_match_within_chunk(self, cut_text, CutFilter):
        text, truncated = self.run(CutFilter, ["abc<END>zzz"], ["<END>"])
        assert truncated and text == "abc<END>"

    def test_match_at_stream_end_is_not_truncation(self, cut_text, CutFilter):
        text, truncated = self.run(CutFilter, ["abc<E", "ND>"], ["<END>"])
        assert not truncated and text == "abc<END>"

    def test_trailing_whitespace_is_not_truncation(self, cut_text, CutFilter):
        text, truncated = self.run(CutFilter, ["abc<END>", "\n", "  \n"], ["<END>"])
        assert not truncated and text == "abc<END>"

    def test_whitespace_then_content_is_truncation(self, cut_text, CutFilter):
        # タグの後ろにいったん改行、そのあと偽の続き -> 打ち切り
        text, truncated = self.run(CutFilter, ["abc<END>", "\n\n", "FAKE RESULT"], ["<END>"])
        assert truncated and text == "abc<END>"

    def test_match_across_chunks_then_content(self, cut_text, CutFilter):
        text, truncated = self.run(CutFilter, ["abc<E", "ND>zzz"], ["<END>"])
        assert truncated and text == "abc<END>"

    def test_match_across_many_chunks(self, cut_text, CutFilter):
        text, truncated = self.run(CutFilter, list("abc<END>zzz"), ["<END>"])
        assert truncated and text == "abc<END>"

    def test_no_match_flushes_everything(self, cut_text, CutFilter):
        text, truncated = self.run(CutFilter, ["hello ", "world"], ["<END>"])
        assert not truncated and text == "hello world"

    def test_after_match_emits_nothing(self, cut_text, CutFilter):
        f = CutFilter(["<END>"])
        f.feed("x<END>")
        assert f.matched and not f.truncating
        assert f.feed("\n") == ""       # 空白: まだ打ち切りではない
        assert not f.truncating
        assert f.feed("more") == ""     # 中身: ここで打ち切り確定
        assert f.truncating

    def test_no_cuts_passthrough(self, cut_text, CutFilter):
        f = CutFilter(None)
        assert f.feed("anything") == "anything"
        assert not f.truncating

    def test_single_char_cut(self, cut_text, CutFilter):
        text, truncated = self.run(CutFilter, ["ab", "c", "def"], ["c"])
        assert truncated and text == "abc"

    def test_multiple_cuts(self, cut_text, CutFilter):
        text, truncated = self.run(
            CutFilter, ["a<yield ", "/>b<finish />"], ["<finish />", "<yield />"]
        )
        assert truncated and text == "a<yield />"

    def test_prefix_overlap_cuts_normal_end(self, cut_text, CutFilter):
        # 接頭辞関係の cut を両方指定しても、正常終了は打ち切りにならない
        text, truncated = self.run(CutFilter, ["done<yield />"], ["<yield", "<yield />"])
        assert not truncated and text == "done<yield />"
