# claude-api-server

Claude Code CLI (サブスクリプション課金) を **OpenAI Chat Completions 互換 API** として中継するローカルサーバー。

従量課金の Claude API を使わずに、月額サブスクリプションの枠内で OpenAI 互換クライアントから Claude を呼べるようにする。

> **注意:** ローカル・個人利用を前提としている。認証は無く、5時間レート制限は全クライアントで共有される。Claude Code のサブスクリプションは対話的なコーディング利用を想定したもので、任意のアプリの API バックエンドとして常用するのは本来の想定から外れる。規模を広げるなら Claude API を使うほうが安全。

## 起動

同じものが2つの形で入っている。機能は同一。

### 単一ファイル版(配布用)

`claude_api_server.py` だけコピーすれば動く。

```bash
pip install fastapi uvicorn
python3 claude_api_server.py --port 8811
```

### モジュール版(開発用)

```bash
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8811
```

**必ずワーカー1本で動かすこと。** セッション対応表もロックもプロセス内メモリにあるため、
`--workers 2` 以上にするとワーカー間で共有されず、直列化が壊れキャッシュヒット率も落ちる
(単一ファイル版は常に1本。モジュール版もデフォルトは1なので、明示的に増やさなければ問題ない)。

## エンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI 互換。`stream: true` で SSE |
| `GET` | `/v1/models` | 利用可能なモデル一覧 |
| `POST` | `/chat` | 素の入出力(OpenAI 互換より前から使っている簡易形式) |
| `GET` | `/health` | 稼働確認 + セッション引き当ての統計 |
| `GET` | `/usage` | サブスクリプションの使用状況(セッション・週次制限の消費率) |

### `/usage` について

claude CLI の `/usage` スラッシュコマンドを headless(`-p`)で実行した結果を返す。

- `limits` にパース済みの消費率(`window` / `model` / `used_percent` / `resets`)、
  `raw` にレポート全文が入る。表記が変わってパースに失敗しても `raw` は返るので、
  クライアントはそちらへフォールバックできる
- モデル呼び出しは発生せず、トークンも消費しない
- 起動に数秒かかるため結果を `CLAUDE_API_USAGE_CACHE_TTL` 秒(既定60)キャッシュする。
  `?refresh=1` で取り直し
- `-p` で `/usage` が動くのはドキュメント化されていない挙動(claude 2.1.237 で確認)。
  CLI の更新で壊れる可能性がある


```bash
curl -s http://127.0.0.1:8811/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"日本の首都は?"}]}'
```

## 対応範囲

**動く**

- 非ストリーミング / ストリーミング(SSE)
- `content` は文字列でも content-parts 配列でも可
- 画像・ファイル添付(data URI をディスクに落とし、Read ツール経由で読ませる)
- `usage`(プロンプトキャッシュの内訳込み)
- 思考レベル: `reasoning_effort`(OpenAI 互換名)または `effort`。
  値は `low` / `medium` / `high` / `xhigh` / `max`(`minimal` は `low` に読み替え)。
  `claude --effort` にそのまま渡る。両方指定時は `effort` が優先。
  レベルはセッションキーに含まれるため、途中で変えるとフル送信に落ちる
- CORS(ブラウザから直接叩ける)

**動かない**

| 項目 | 挙動 |
|---|---|
| `tools` / function calling | **黙って無視される**(エラーにならないので注意) |
| `temperature` / `max_tokens` / `n` / `response_format` など | 黙って無視(`claude -p` に対応するフラグが無い) |
| `role: "tool"` のメッセージ | 422 |
| 末尾が assistant のメッセージ列(prefill) | 400 |
| `/v1/embeddings`, `/v1/completions` | 未実装 |
| `stop`(OpenAI 標準) | 黙って無視。代わりに拡張 `x_cut_after` を使う(下記) |

### 拡張: `x_cut_after`(応答の切り詰め)

```json
{"model": "sonnet", "messages": [...], "x_cut_after": ["<yield />", "<finish />"]}
```

指定した文字列のいずれかが応答に最初に現れた「直後」で応答を切る。
**OpenAI の `stop` と意味論が違う**: `stop` は一致文字列を含めないが、こちらは
一致文字列を**残して**その直後で切る(終端タグをパーサが必要とするクライアント向け)。

「打ち切り」扱いになるのは**切り位置の後ろに空白以外の中身があったとき**だけ。
応答が切り文字列そのもので終わる(終端タグで正常に終わるクライアントの通常ターン)
場合は一致しても打ち切りではなく、セッションは通常どおり登録され、`usage` も返る。

本物の打ち切りが起きたターンは**セッションを登録せず破棄する**(セッション側には
クライアントが知らない内容が残ってしまうため)。次のリクエストは自動的にフル送信
(miss)へ落ちる。これは想定どおりの代償で、`/health` の `truncated` で回数を
確認できる。ストリーミングでは切り位置以降は SSE に流れず、打ち切りが確定した
時点で生成も止まる(このときだけ `usage` は取得できず 0 になる)。
| 認証 | 無し |

モデル名は `sonnet` / `opus` / `claude-sonnet-5` など claude CLI が解釈できるもの。
`gpt-4o` 等を固定で送るクライアントはエラーになる。

## セッション再利用によるキャッシュ最適化

クライアントは **毎回全会話履歴を送るステートレス方式のままでよい**。サーバー側だけがセッションを持つ。

### なぜ必要か

Anthropic のプロンプトキャッシュはプレフィックス一致で、キャッシュ境界はコンテンツブロック単位。
全履歴を1つのテキストとして毎回渡すと、会話が1ターン伸びるだけでブロック全体が別物になり、
履歴全体がキャッシュ**書き込み**(通常入力の 1.25〜2倍課金相当)として再送される。

実測(会話が伸びていくパターン、1ターンあたり):

| 方式 | cache_write | cache_read |
|---|---|---|
| 毎回フル送信 | 13,571 | 1,070 |
| セッション再利用 | **21** | **14,421** |

### 仕組み

1. リクエストの全履歴をレンダリングし、末尾から遡ってプレフィックスのハッシュで既存セッションを探す
2. ヒットしたら `--resume <session_id>` で **差分ターンだけ** を送る
3. 応答後、「今回の全履歴 + 今回返した応答」のハッシュ → session_id を登録する
4. ミスなら従来どおり全履歴を送り、新しいセッションとして登録する

**ハッシュのキーには「サーバーが返した assistant 応答」も含める。** これが安全弁で、
クライアントが応答を加工して保存していた場合はキーが一致せず、自動的にフル送信へ落ちる。
履歴が食い違ったまま会話が進むことはない。

`GET /health` の `hit_rate` で実績を確認できる。低い場合はクライアントが応答を加工している。

### 同時アクセス

- **別会話** は並行に実行される
- **同一会話** は会話単位のロックで直列化される(引き当て→実行→登録をまとめて保護)

## 設定 (環境変数)

| 変数 | 既定値 | 説明 |
|---|---|---|
| `CLAUDE_BIN` | `claude` | claude CLI のパス |
| `CLAUDE_API_WORKDIR` | `~/.claude-api-workdir` | claude を実行する作業ディレクトリ |
| `CLAUDE_API_SESSION_REUSE` | `1` | `0` でセッション再利用を無効化(毎回フル送信) |
| `CLAUDE_API_MAX_SESSIONS` | `64` | 保持するセッション数(LRU) |
| `CLAUDE_API_SESSION_TTL` | `86400` | セッションの寿命(秒) |
| `CLAUDE_API_MODELS` | 下記 | `/v1/models` が返す一覧(カンマ区切り) |

既定のモデル一覧: `sonnet, opus, haiku, claude-sonnet-5, claude-opus-5, claude-haiku-4-5`

## 構成

| ファイル | 役割 |
|---|---|
| `claude_api_server.py` | **単一ファイル版**。これ1つで完結する(配布用) |
| `server.py` | モジュール版: FastAPI アプリ、エンドポイント、SSE 整形 |
| `claude.py` | claude CLI の起動と出力の解釈 |
| `conversation.py` | メッセージのレンダリング、会話キー、実行プランの決定 |
| `sessions.py` | セッション対応表(LRU + TTL)、会話ロック |
| `config.py` | 環境変数による設定 |

単一ファイル版とモジュール版は独立に保守されるため乖離しうる。
`tests/test_single_file_parity.py` が両者の出力一致を検証しており、
ロジックを変えたら片方だけ直した時点でテストが落ちる。

### 実装上の要点

- **プロンプトは stdin 経由**。CLI 引数には Linux の1引数あたりの長さ上限
  (`MAX_ARG_STRLEN`, 通常128KB)があり、会話履歴を載せると
  `OSError: [Errno 7] Argument list too long` になる。system prompt も同じ理由で
  `--system-prompt-file` を使う。
- **`--bare` は使わない**。bare は `ANTHROPIC_API_KEY` 認証を強制し、
  OAuth(サブスク課金)を無効化してしまう。
- **添付ファイルの保存パスは内容のハッシュ由来**。同じ添付が常に同じパスに落ちるので
  会話ハッシュが安定し、セッション再利用が効く。
- **stream-json の読み取り上限を拡張**。1行に添付の base64 が載ることがあり、
  asyncio の `readline()` 既定上限(64KB)では足りない。

## テスト

claude を起動しない純粋ロジックの単体テスト:

```bash
pip install pytest
python3 -m pytest tests -q
```
