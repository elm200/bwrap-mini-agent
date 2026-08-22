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
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DONE_MARKER = "TASK_COMPLETE"

LOG_FILE = Path("mini_agent.log")  # ハーネス<->LLM間の生のやり取りを追記していくログ

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
# ログ — ハーネス<->LLM間のリクエスト/レスポンスをなるべく忠実に記録する
# --------------------------------------------------------------------------

def log_exchange(direction: str, body, meta: str = "") -> None:
    ts = datetime.now().astimezone().isoformat(timespec="milliseconds")
    header = f"[{ts}] {direction}" + (f" ({meta})" if meta else "")
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, indent=2)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{'=' * 80}\n{header}\n{'=' * 80}\n{text}\n\n")


# --------------------------------------------------------------------------
# OpenRouterクライアント(OpenAI互換のchat completions)
# --------------------------------------------------------------------------

def call_openrouter(messages: list[dict], model: str, api_key: str) -> dict:
    request_payload = {"model": model, "messages": messages}
    log_exchange("REQUEST", request_payload)

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(request_payload).encode("utf-8"),
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
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_body = resp.read().decode("utf-8")
            data = json.loads(raw_body)
        elapsed = time.perf_counter() - start
    except urllib.error.HTTPError as e:
        raw_body = e.read().decode("utf-8", errors="replace")
        try:
            error_data = json.loads(raw_body)
        except json.JSONDecodeError:
            error_data = raw_body
        log_exchange("RESPONSE", error_data, meta=f"HTTP {e.code} エラー")
        raise RuntimeError(f"OpenRouterへのリクエストが失敗しました (HTTP {e.code}): {raw_body}") from e

    log_exchange("RESPONSE", data, meta=f"{elapsed:.2f}秒")
    return {
        "content": data["choices"][0]["message"]["content"],
        "usage": data.get("usage") or {},
        "elapsed": elapsed,
    }


def format_usage(usage: dict, elapsed: float) -> str:
    parts = []
    if (p := usage.get("prompt_tokens")) is not None:
        parts.append(f"入力:{p}tok")
    if (c := usage.get("completion_tokens")) is not None:
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        parts.append(f"出力:{c}tok" + (f"(思考:{reasoning}tok)" if reasoning else ""))
    if (cost := usage.get("cost")) is not None:
        parts.append(f"費用:{cost * 100:.4f}セント")
    parts.append(f"所要:{elapsed:.2f}秒")
    return "[使用量] " + " / ".join(parts)


def format_totals(totals: dict, elapsed: float, steps: int) -> str:
    parts = [
        f"{steps}ステップ",
        f"入力:{totals['prompt_tokens']}tok",
        f"出力:{totals['completion_tokens']}tok"
        + (f"(思考:{totals['reasoning_tokens']}tok)" if totals["reasoning_tokens"] else ""),
        f"費用:{totals['cost'] * 100:.4f}セント",
        f"LLM応答時間合計:{elapsed:.2f}秒",
    ]
    return "[累計] " + " / ".join(parts)


# --------------------------------------------------------------------------
# エージェントループ — minisweagent/agents/default.py を元に改変
# --------------------------------------------------------------------------

def run_agent(task: str, model: str, api_key: str, max_steps: int, sandbox_root: Path) -> None:
    log_exchange("TASK開始", {"task": task, "model": model, "max_steps": max_steps, "sandbox_root": str(sandbox_root)})

    sandbox = Sandbox(sandbox_root)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTANCE_TEMPLATE.format(task=task, cwd=sandbox.root)},
    ]

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost": 0.0}
    total_elapsed = 0.0

    for step in range(1, max_steps + 1):
        print(f"\n=== step {step}/{max_steps} ===")
        call = call_openrouter(messages, model, api_key)
        reply = call["content"]
        usage = call["usage"]
        messages.append({"role": "assistant", "content": reply})
        print(reply.strip())
        print(format_usage(usage, call["elapsed"]))

        totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
        totals["completion_tokens"] += usage.get("completion_tokens") or 0
        totals["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        totals["cost"] += usage.get("cost") or 0.0
        total_elapsed += call["elapsed"]

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
            print(format_totals(totals, total_elapsed, step))
            return

        observation = f"<output>\n{output}\n</output>\n<returncode>{result['returncode']}</returncode>"
        messages.append({"role": "user", "content": observation})

    print(f"\n⚠️  {max_steps}ステップ経過しましたが完了しませんでした。")
    print(format_totals(totals, total_elapsed, max_steps))


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
