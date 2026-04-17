#!/usr/bin/env python3
"""Generate AscendC kernels using Claude Code CLI (single-shot, no tools).

Two test scenarios controlled by --with-best-practices:
  1. Baseline   : use the standard prompt from PROMPT_REGISTRY.
  2. With BP    : prepend ascendc-api-best-practices.md to the prompt.

A constraint block is always appended to tell Claude Code:
  - generate once, do not invoke any tool,
  - do not attempt on-device compile/test,
  - emit the full answer to stdout only.
"""

import os
import sys
import subprocess
import argparse
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_and_write import generate_prompt
from dataset import dataset
from config import temperature, top_p


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BEST_PRACTICES_PATH = os.path.join(REPO_ROOT, 'ascendc-api-best-practices.md')

CONSTRAINT_SUFFIX = """

---
重要约束（必须严格遵守）：
1. 你只能单次直接生成完整答案，不能多次调用任何工具、检索或重新查询。
2. 你不能进行上板/编译测试，只能依赖自身的 LLM 能力和上下文进行推理。
3. 请将完整生成内容直接输出到 stdout，不要保存或写入任何文件。
"""


def load_best_practices():
    with open(BEST_PRACTICES_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def build_prompt(language, strategy, op, best_practices=None):
    base = generate_prompt(language, strategy, op)
    parts = []
    if best_practices:
        parts.append("以下是 AscendC API 最佳实践参考，请在生成时严格遵循其中的规范：\n\n")
        parts.append(best_practices)
        parts.append("\n\n---\n\n")
    parts.append(base)
    parts.append(CONSTRAINT_SUFFIX)
    return ''.join(parts)


def generate_with_claude_code(prompt, out_dir, op, timeout=300, disable_skills=False):
    """Invoke claude CLI in print mode with all tools disabled."""
    out_path = os.path.join(out_dir, f'{op}.txt')
    if os.path.exists(out_path):
        print(f"[INFO] Already generated at {out_path}, skip")
        return

    claude_bin = shutil.which("claude") or "/usr/bin/claude"
    cmd = [
        claude_bin,
        "-p",
        "--output-format", "text",
        "--tools", "",
    ]
    if disable_skills:
        cmd.append("--disable-slash-commands")

    print(f"[INFO] Generating {op} via Claude Code CLI (prompt_len={len(prompt)})...")
    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Timeout ({timeout}s) for {op}")
        return

    elapsed = time.time() - start
    print(f"[INFO] {op} completed in {elapsed:.1f}s, "
          f"exit_code={proc.returncode}, output_len={len(proc.stdout)}")

    if proc.returncode != 0:
        print(f"[WARN] Non-zero exit for {op}: {proc.stderr[:500]}")

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


def resolve_ops(args):
    if args.ops:
        return list(args.ops)
    cats = set(args.categories)
    ops = [op for op, meta in dataset.items() if meta.get("category") in cats]
    if not ops:
        raise SystemExit(f"[ERROR] No ops found for categories={args.categories}")
    return ops


def main():
    parser = argparse.ArgumentParser(
        description="Generate AscendC kernels using Claude Code CLI")
    parser.add_argument('--model-name', type=str, default='claude-code',
                        help='Output directory model name (use different values to separate runs)')
    parser.add_argument('--strategy', type=str, default='add_shot')
    parser.add_argument('--categories', nargs='+', default=['fuse'],
                        help='Dataset categories to generate (default: fuse)')
    parser.add_argument('--ops', nargs='+', default=None,
                        help='Explicit operator list, overrides --categories')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout per operator in seconds')
    parser.add_argument('--disable-skills', action='store_true',
                        help='Pass --disable-slash-commands to claude CLI')
    parser.add_argument('--with-best-practices', action='store_true',
                        help='Prepend ascendc-api-best-practices.md to the prompt')

    args = parser.parse_args()
    language = 'ascendc'
    ops = resolve_ops(args)

    best_practices = load_best_practices() if args.with_best_practices else None

    print(f"Model name      : {args.model_name}")
    print(f"Strategy        : {args.strategy}")
    print(f"Categories      : {args.categories}")
    print(f"Ops (count={len(ops)}): {ops[:5]}{'...' if len(ops) > 5 else ''}")
    print(f"Disable skills  : {args.disable_skills}")
    print(f"With BP         : {args.with_best_practices}"
          + (f" ({len(best_practices)} chars)" if best_practices else ""))

    for run in range(args.runs):
        out_dir = f'output/{language}/{args.strategy}/{temperature}-{top_p}/{args.model_name}/run{run}'
        os.makedirs(out_dir, exist_ok=True)

        for op in ops:
            try:
                prompt = build_prompt(language, args.strategy, op, best_practices=best_practices)
                generate_with_claude_code(prompt, out_dir, op,
                                          timeout=args.timeout,
                                          disable_skills=args.disable_skills)
            except Exception as e:
                print(f"[ERROR] Failed for {op}: {e}")


if __name__ == '__main__':
    main()
