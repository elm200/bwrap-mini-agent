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

SYSTEM_PROMPT = f"""You are a helpful assistant that can interact with a computer via bash.

Rules:
- Your response must contain exactly ONE bash code block with ONE command
  (or several commands chained with && / ||).
- Include a short THOUGHT explaining your reasoning before the command.
- You can ONLY read and write files under the current directory (which is
  already the sandbox root — do not try to `cd` out of it, it will fail).
- There is NO network access. Do not attempt to curl/wget/pip install
  anything that requires the internet — it will fail.
- When the task is fully done, run exactly:
    echo {DONE_MARKER} && echo "<one-line summary of what you did>"
  as your final command. Do not combine it with anything else.

Format your response as shown below.

THOUGHT: <your reasoning>
```bash
<your single command>
```
"""

INSTANCE_TEMPLATE = """Task: {task}

Working directory: {cwd} (this IS the sandbox root)
Remember: exactly one bash command per response, inside a ```bash block.
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
            return {"output": out + f"\n[timed out after {self.timeout}s]", "returncode": -1}
        except Exception as e:  # noqa: BLE001 — サンドボックス起動失敗などを捕捉
            return {"output": f"[sandbox error] {e}", "returncode": -1}


# --------------------------------------------------------------------------
# 行動のパース — minisweagent/models/utils/actions_text.py を元に改変
# --------------------------------------------------------------------------

class FormatError(Exception):
    pass


def parse_action(content: str) -> str:
    matches = [m.strip() for m in ACTION_RE.findall(content)]
    if len(matches) != 1:
        raise FormatError(
            f"Expected exactly 1 ```bash``` block, found {len(matches)}. "
            "Reminder: respond with exactly one bash code block."
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
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from e
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
            print(f"[format error] {e}")
            messages.append({"role": "user", "content": f"FORMAT ERROR: {e}"})
            continue

        result = sandbox.execute(command)
        output = result["output"]
        print(f"--- output (exit {result['returncode']}) ---\n{output}")

        first_line = output.lstrip().splitlines()[0].strip() if output.strip() else ""
        if first_line == DONE_MARKER and result["returncode"] == 0:
            summary = "\n".join(output.lstrip().splitlines()[1:]).strip()
            print(f"\n✅ done: {summary}")
            return

        observation = f"<output>\n{output}\n</output>\n<returncode>{result['returncode']}</returncode>"
        messages.append({"role": "user", "content": observation})

    print(f"\n⚠️  stopped after {max_steps} steps without completion.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file bwrap-sandboxed coding agent (OpenRouter).")
    parser.add_argument("task", help="task description for the agent")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-steps", type=int, default=15, help="max loop iterations (default: 15)")
    parser.add_argument(
        "--sandbox-dir",
        default=SANDBOX_DIRNAME,
        help=f"sandbox directory, created if missing (default: ./{SANDBOX_DIRNAME})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("error: set OPENROUTER_API_KEY in your environment first.")

    if not shutil.which("bwrap"):
        sys.exit("error: 'bwrap' (bubblewrap) not found. Install it first (e.g. apt install bubblewrap).")

    sandbox_root = Path(args.sandbox_dir).resolve()
    run_agent(args.task, args.model, api_key, args.max_steps, sandbox_root)


if __name__ == "__main__":
    main()
