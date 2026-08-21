#!/usr/bin/env python3
"""
mini_agent.py — 1ファイル・bwrapサンドボックス方式のコーディングエージェント。

設計はmini-swe-agent (https://github.com/SWE-agent/mini-swe-agent) から借用:
  - エージェントループ:      agents/default.py
  - bubblewrapサンドボックス: environments/extra/bubblewrap.py
  - 正規表現による行動パース: models/utils/actions_text.py

mini-swe-agentとの違い:
  - すべてを1ファイルに収め、必要最小限に削ぎ落としている。
  - エージェントが触れるのは固定の "sandbox/" サブディレクトリのみ(bwrap内で
    読み書き可能としてバインド)。それ以外のファイルシステムは読み取り専用で
    マウントされ、サンドボックス内のネットワークは遮断される (--unshare-net)。
  - モデルプロバイダーはOpenRouterのOpenAI互換エンドポイントに固定。
  - デフォルトモデル: google/gemini-2.5-flash-lite

使い方:
    export OPENROUTER_API_KEY=sk-or-...
    python3 mini_agent.py "sandbox/haiku.py に俳句ジェネレーターを書いて実行して"

    # モデルやステップ数上限を上書きする場合
    python3 mini_agent.py --model openai/gpt-4o-mini --max-steps 20 "タスク内容"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DONE_MARKER = "TASK_COMPLETE"

SANDBOX_DIRNAME = "sandbox"  # エージェントが読み書きできる唯一のディレクトリ

SYSTEM_PROMPT = f"""あなたはbash経由でコンピュータを操作できる有能なアシスタントです。

ルール:
- 回答には、1つのコマンド(または && / || で連結した複数コマンド)を含む
  bashコードブロックをちょうど1つだけ含めること。
- コマンドの前に、判断理由を短く説明するTHOUGHTを書くこと。
- 読み書きできるのは現在のディレクトリ以下のみ(すでにサンドボックスの
  ルートになっている。`cd` で外に出ようとしても失敗する)。
- ネットワークアクセスはできない。インターネット接続が必要なcurl/wget/
  pip installなどは実行しないこと(失敗する)。
- タスクが完全に完了したら、最後のコマンドとして必ず次を単体で実行すること:
    echo {DONE_MARKER} && echo "<行った内容の1行要約>"
  他のコマンドと組み合わせないこと。

回答は以下の形式に従うこと。

THOUGHT: <判断理由>
```bash
<単一のコマンド>
```
"""

INSTANCE_TEMPLATE = """タスク: {task}

作業ディレクトリ: {cwd} (これがサンドボックスのルートです)
忘れずに: 1回の回答につきbashコマンドは1つだけ、```bash ブロック内に書くこと。
"""

ACTION_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------
# サンドボックス実行(bwrap)— minisweagent/environments/extra/bubblewrap.py
# を元に改変
# --------------------------------------------------------------------------

class Sandbox:
    def __init__(self, root: Path, timeout: int = 30):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.executable = shutil.which("bwrap") or "bwrap"

    def _bwrap_cmd(self, command: str) -> list[str]:
        cmd = [
            self.executable,
            "--unshare-user-try",
            "--unshare-net",  # <- サンドボックス内のネットワークを遮断
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
        ]
        if Path("/lib64").exists():
            cmd += ["--ro-bind", "/lib64", "/lib64"]
        cmd += [
            "--ro-bind", "/etc", "/etc",
            "--tmpfs", "/tmp",
            "--proc", "/proc",
            "--dev", "/dev",
            "--new-session",
            "--setenv", "PATH", "/usr/local/bin:/usr/sbin:/usr/bin:/bin",
            # 書き込み可能なのはsandboxルートのみ。基本OSディレクトリ以外で
            # ホストのファイルシステムからバインドされるのもこれだけ
            "--bind", str(self.root), str(self.root),
            "--chdir", str(self.root),
            "bash", "-c", command,
        ]
        return cmd

    def execute(self, command: str) -> dict:
        try:
            result = subprocess.run(
                self._bwrap_cmd(command),
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired as e:
            out = e.output or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            return {"output": out + f"\n[{self.timeout}秒でタイムアウトしました]", "returncode": -1}
        except Exception as e:  # noqa: BLE001 — サンドボックス起動失敗などを捕捉
            return {"output": f"[サンドボックスエラー] {e}", "returncode": -1}


# --------------------------------------------------------------------------
# 行動のパース — minisweagent/models/utils/actions_text.py を元に改変
# --------------------------------------------------------------------------

class FormatError(Exception):
    pass


def parse_action(content: str) -> str:
    matches = [m.strip() for m in ACTION_RE.findall(content)]
    if len(matches) != 1:
        raise FormatError(
            f"```bash```ブロックはちょうど1つである必要がありますが、{len(matches)}個見つかりました。"
            "リマインダー: bashコードブロックを1つだけ含めて回答してください。"
        )
    return matches[0]


# --------------------------------------------------------------------------
# OpenRouterクライアント(OpenAI互換のchat completions)
# --------------------------------------------------------------------------

def call_openrouter(messages: list[dict], model: str, api_key: str) -> str:
    payload = json.dumps({"model": model, "messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # 必須ではないがOpenRouterが推奨しているヘッダー:
            "HTTP-Referer": "https://local.mini-agent",
            "X-Title": "mini_agent.py",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouterへのリクエストが失敗しました (HTTP {e.code}): {body}") from e
    return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# エージェントループ — minisweagent/agents/default.py を元に改変
# --------------------------------------------------------------------------

def run_agent(task: str, model: str, api_key: str, max_steps: int, sandbox_root: Path) -> None:
    sandbox = Sandbox(sandbox_root)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTANCE_TEMPLATE.format(task=task, cwd=sandbox.root)},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n=== step {step}/{max_steps} ===")
        reply = call_openrouter(messages, model, api_key)
        messages.append({"role": "assistant", "content": reply})
        print(reply.strip())

        try:
            command = parse_action(reply)
        except FormatError as e:
            print(f"[フォーマットエラー] {e}")
            messages.append({"role": "user", "content": f"フォーマットエラー: {e}"})
            continue

        result = sandbox.execute(command)
        output = result["output"]
        print(f"--- 出力 (終了コード {result['returncode']}) ---\n{output}")

        first_line = output.lstrip().splitlines()[0].strip() if output.strip() else ""
        if first_line == DONE_MARKER and result["returncode"] == 0:
            summary = "\n".join(output.lstrip().splitlines()[1:]).strip()
            print(f"\n✅ 完了: {summary}")
            return

        observation = f"<output>\n{output}\n</output>\n<returncode>{result['returncode']}</returncode>"
        messages.append({"role": "user", "content": observation})

    print(f"\n⚠️  {max_steps}ステップ経過しましたが完了しませんでした。")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="1ファイル・bwrapサンドボックス方式のコーディングエージェント(OpenRouter使用)。")
    parser.add_argument("task", help="エージェントに与えるタスクの説明")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouterのモデル (デフォルト: {DEFAULT_MODEL})")
    parser.add_argument("--max-steps", type=int, default=15, help="ループの最大反復回数 (デフォルト: 15)")
    parser.add_argument(
        "--sandbox-dir",
        default=SANDBOX_DIRNAME,
        help=f"サンドボックスディレクトリ。存在しなければ作成される (デフォルト: ./{SANDBOX_DIRNAME})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("エラー: 先に環境変数 OPENROUTER_API_KEY を設定してください。")

    if not shutil.which("bwrap"):
        sys.exit("エラー: 'bwrap' (bubblewrap) が見つかりません。先にインストールしてください (例: apt install bubblewrap)。")

    sandbox_root = Path(args.sandbox_dir).resolve()
    run_agent(args.task, args.model, api_key, args.max_steps, sandbox_root)


if __name__ == "__main__":
    main()
