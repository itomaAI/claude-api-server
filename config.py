"""環境変数による設定と、実行時に使うパス。"""

import os

# --- claude CLI ---
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")

# claude を実行する作業ディレクトリ。空のディレクトリを使い、他プロジェクトの
# CLAUDE.md が意図せず読み込まれないようにする。
WORKDIR = os.environ.get(
    "CLAUDE_API_WORKDIR", os.path.expanduser("~/.claude-api-workdir")
)
ATTACH_DIR = os.path.join(WORKDIR, "attachments")
TMP_DIR = os.path.join(WORKDIR, "tmp")

# --- セッション再利用(プロンプトキャッシュ最適化) ---
# 0 で無効化し、毎回フル送信する従来動作に戻る。
SESSION_REUSE = os.environ.get("CLAUDE_API_SESSION_REUSE", "1") != "0"
MAX_SESSIONS = int(os.environ.get("CLAUDE_API_MAX_SESSIONS", "64"))
SESSION_TTL = float(os.environ.get("CLAUDE_API_SESSION_TTL", "86400"))

# 直近何ターン分さかのぼってプレフィックス一致を探すか。通常は1〜2ターンで当たる
# (新しい user 発言のみ / ツール出力+user 発言 など)。
LOOKBACK_TURNS = 8

# 同時に保持する会話ロックの上限。
MAX_CONV_LOCKS = 256

# stream-json の1行は Read ツール経由の base64 を含みうるので、asyncio の
# readline() 既定上限(64KB)では足りない。クライアント側の添付上限に余裕を持たせた値。
STREAM_LINE_LIMIT = 100 * 1024 * 1024

# --- /v1/models で返すモデル ---
# claude CLI にモデル列挙コマンドは無いため定数で持つ。実際に使えるかは
# 契約プランに依存するので、環境変数でカンマ区切り上書きできるようにしてある。
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


def ensure_dirs() -> None:
    os.makedirs(ATTACH_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
