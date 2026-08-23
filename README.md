# mini_agent.py

bubblewrap(bwrap)でサンドボックス化した環境の中で、LLM(OpenRouter経由)にbashコマンドを1つずつ実行させる、1ファイルのミニコーディングエージェント。

[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) にアイディアを得たもので、エージェントループ・bubblewrapサンドボックス・行動パースの設計を借用している。すべてを1ファイルに収め、必要最小限に削ぎ落としている点、サンドボックスをホームディレクトリ内の固定サブディレクトリに限定している点、モデルプロバイダーをOpenRouterに固定している点が本家との主な違い。

## 主な機能

- OpenRouterのchat completions APIを呼び出し、`THOUGHT: ...` + ` ```bash ``` `形式の応答をパースして、コマンドを1つずつ実行するエージェントループ
- bubblewrapで隔離されたサンドボックス内でコマンドを実行。書き込み可能なのは指定したサンドボックスディレクトリ(既定: `./sandbox`)のみで、それ以外のファイルシステムは読み取り専用、ネットワークアクセスは遮断(`--unshare-net`)される
- ハーネスとLLM間の生のリクエスト/レスポンスを`mini_agent.log`に逐次記録
- ステップごと・累計のトークン数/費用/所要時間をターミナルに表示
- `--model` / `--max-steps` / `--sandbox-dir` / `--no-reasoning` オプションで挙動をカスタマイズ可能
- 単体でも同等のサンドボックスを起動できる`bwrap.sh`を同梱

## 技術スタック

- Python 3(標準ライブラリのみ、外部依存なし)
- bubblewrap(bwrap) — Linuxのユーザー名前空間サンドボックス
- OpenRouter API(OpenAI互換のchat completions、既定モデル: `google/gemini-2.5-flash-lite`)

## セットアップ

前提: Python 3、bubblewrap(`apt install bubblewrap`)、OpenRouterのAPIキー

```bash
export OPENROUTER_API_KEY=sk-or-...
python3 mini_agent.py "haiku.py に俳句ジェネレーターを書いて実行して"
```

主なオプション:

- `--model` : モデル指定(既定: `google/gemini-2.5-flash-lite`)
- `--max-steps` : ループの最大反復回数(既定: 15)
- `--sandbox-dir` : サンドボックスディレクトリ(既定: `./sandbox`。存在しなければ作成される)
- `--no-reasoning` : `reasoning: {enabled: false}` を送り思考を無効化する(対応モデルのみ)

環境変数:

- `OPENROUTER_API_KEY`(必須) — OpenRouterのAPIキー

## 使い方

実行するとエージェントがタスクを受け取り、各ステップで以下を繰り返す。

1. LLMが`THOUGHT`(判断理由)と、bashコードブロック1つに包んだコマンドを1つ生成
2. そのコマンドをbwrapサンドボックス内で実行し、標準出力/終了コードを観察
3. 観察結果をLLMに渡して次のステップへ

出力中に`TASK_COMPLETE`とだけ一致する行が現れると完了とみなし、その後に続く行を作業の要約として表示してループを終了する。各ステップの使用量(`[使用量]`行)、完了時・中断時の累計(`[累計]`行)がターミナルに表示される。ハーネスとLLM間の生のやり取りは`mini_agent.log`に追記され続ける。

エージェントが触れられるファイルシステムはbwrapで隔離されており、書き込み可能なのはサンドボックスディレクトリ以下のみ。それ以外は読み取り専用でマウントされ、サンドボックス内からのネットワークアクセスもできない。

## License

MIT License. 詳細は[LICENSE](LICENSE)を参照。
