## claude code算子生成能力测试代码生成

这个代码需要完成两个测试，不同类别以参数的形式控制
1. 调用claude code生成reference里面fuse类算子的生成编译准确度

2. 调用claude code生成reference里面fuse类算子的生成编译准确度，但是prompt里面需要增加./ascendc-api-best-practices.md里面的内容作为上线

注意：
prompt里面要增加一些给claude code的限制：
- 只能单次直接生成，不能多次调用
- 不能上板测试，只能通过自己的LLM能力和上下文生成对应的txt文件

代码参考

```python
#!/usr/bin/env python3
"""Generate AscendC kernels using Claude Code CLI (single-shot, no tools)."""

import os
import sys
import subprocess
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_and_write import generate_prompt
from dataset import dataset
from config import temperature, top_p


def generate_with_claude_code(prompt, out_dir, op, timeout=300, disable_skills=False):
    """Invoke claude CLI in print mode with all tools disabled."""
    out_path = os.path.join(out_dir, f'{op}.txt')
    if os.path.exists(out_path):
        print(f"[INFO] Already generated at {out_path}, skip")
        return

    import shutil
    claude_bin = shutil.which("claude") or "/usr/bin/claude"
    cmd = [
        claude_bin,
        "-p",
        "--output-format", "text",
        "--tools", "",
    ]
    if disable_skills:
        cmd.append("--disable-slash-commands")

    print(f"[INFO] Generating {op} via Claude Code CLI...")
    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Timeout ({timeout}s) for {op}")
        return

    elapsed = time.time() - start
    print(f"[INFO] {op} completed in {elapsed:.1f}s, "
          f"exit_code={proc.returncode}, output_len={len(proc.stdout)}")

    if proc.returncode != 0:
        print(f"[WARN] Non-zero exit for {op}: {proc.stderr[:500]}")

    # Skip writing rate-limit / auth error placeholders (allows later retry)
    out = (proc.stdout or '').strip()
    rate_limit_markers = [
        "You've hit your limit",
        "Please log in",
        "Rate limit exceeded",
    ]
    if len(out) < 200 or any(m in out for m in rate_limit_markers):
        print(f"[SKIP] {op}: output looks like rate-limit/error ({len(out)} bytes): {out[:80]}")
        return

    with open(out_path, 'w') as f:
        f.write(proc.stdout)


def main():
    parser = argparse.ArgumentParser(
        description="Generate AscendC kernels using Claude Code CLI")
    parser.add_argument('--model-name', type=str, required=True,
                        help='Output directory model name')
    parser.add_argument('--strategy', type=str, default='add_shot')
    parser.add_argument('--ops', nargs='+', required=True,
                        help='Operators to generate')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per operator in seconds')
    parser.add_argument('--disable-skills', action='store_true',
                        help='Disable skills loading via --disable-slash-commands')

    args = parser.parse_args()
    language = 'ascendc'

    print(f"Model name: {args.model_name}")
    print(f"Strategy: {args.strategy}")
    print(f"Operators: {args.ops}")
    print(f"Disable skills: {args.disable_skills}")

    for run in range(args.runs):
        out_dir = f'output/{language}/{args.strategy}/{temperature}-{top_p}/{args.model_name}/run{run}'
        os.makedirs(out_dir, exist_ok=True)

        for op in args.ops:
            prompt = generate_prompt(language, args.strategy, op)
            try:
                generate_with_claude_code(prompt, out_dir, op,
                                          timeout=args.timeout,
                                          disable_skills=args.disable_skills)
            except Exception as e:
                print(f"[ERROR] Failed for {op}: {e}")


if __name__ == '__main__':
    main()
```

让claude code直接生成结果就行，不能调用工具查询，只依赖自身的能力和上下文的能力，生成的内容保存对应的文件到不同的路径下