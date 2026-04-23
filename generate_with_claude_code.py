#!/usr/bin/env python3
"""Generate AscendC kernels using Claude Code CLI.

Three test scenarios:
  1. Baseline        : use the standard prompt from PROMPT_REGISTRY (all tools disabled).
  2. With BP         : prepend ascendc-api-best-practices.md to the prompt (all tools disabled).
  3. With Wiki       : enable Read/Glob/Grep tools for local AscendC Kernel Wiki retrieval.

A constraint block is always appended to tell Claude Code:
  - (baseline/bp) generate once, do not invoke any tool,
  - (wiki) allow local wiki retrieval via Read/Glob/Grep, then generate once,
  - do not attempt on-device compile/test,
  - emit the full answer to stdout only.
"""

import os
import re
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
WIKI_DIR = os.path.join(REPO_ROOT, 'AscendC-Kernel-Wiki', 'wiki')

WIKI_PREAMBLE = f"""以下是本地 AscendC Kernel Wiki 知识库，你可以在生成前通过 Read、Glob、Grep 工具查阅其中的 API 文档和示例代码。

Wiki 本地路径：{WIKI_DIR}

Wiki 目录结构：
```
wiki/
├── SCHEMA.md        # 标签体系、页面类型定义
├── index.md         # 全局索引——查询入口
├── guide/
│   ├── concepts/    # 编程模型、内存层级、流水线同步...
│   ├── api/         # 向量计算、数据搬运、矩阵 API...
│   ├── practice/    # 向量编程、Tiling、双缓冲、融合算子编程...
│   ├── engineering/ # 自定义算子工程、框架适配...
│   └── toolchain/   # msopgen、bisheng、msprof
├── patterns/        # 逐元素、归约、矩阵乘、Tiling 等编程模式
├── operators/       # 860+ 算子，按 nn/math/transformer/cv 分类
└── sources/         # 原始资料摘要
```

Wiki 查询协议（请严格按此步骤检索知识）：

**步骤 1 — 会话定向**
1. 读取 {WIKI_DIR}/index.md — 获取全局索引，了解已有页面和分类结构。
2. 读取 {WIKI_DIR}/SCHEMA.md — 理解标签体系和页面类型。

**步骤 2 — 页面发现**
1. 根据当前算子涉及的操作类型，从 index.md 按关键词匹配候选页面。
2. 编程模式问题（逐元素、归约、矩阵乘等）→ 重点查阅 wiki/patterns/ 下的模式页。
3. 具体算子实现 → 定位 wiki/operators/ 对应分类目录。
4. API 用法（DataCopy、Reduce、Cast、Matmul 等）→ 查阅 wiki/guide/api/ 对应页面。
5. 融合算子开发 → 查阅 wiki/guide/practice/fusion-programming.md。
6. 使用 Grep 在 {WIKI_DIR}/**/*.md 中搜索与当前算子相关的关键词，与 index 结果合并。

**步骤 3 — 深度读取**
1. 读取 top 3~5 候选页面的完整内容。
2. 关注页面中的代码示例、API 约束、参数限制、性能注意事项。
3. 将查阅到的 API 用法、参数限制、最佳实践融入你的内核生成中。

注意：请仅使用 Read、Glob、Grep 工具查阅上述 Wiki 目录中的文件，不要修改任何文件或执行其他工具操作。
"""

OUTPUT_FORMAT_RULES = """
输出格式（最终答复必须严格遵守）：
- 最终答复的第一个非空字符必须是 `project_json_src` 的 `p`。换言之，第一行就是 `project_json_src='''` 开头。
- 禁止在代码之前写任何开场白/总结句，尤其不要以以下任一方式开头：
  "I found...", "Now I have...", "Let me...", "I've gathered...", "Here's the...",
  "I now have...", "好的", "让我", "现在我已经", "根据", "基于".
- 禁止在代码之后追加任何说明、注释、总结或后记。
- 禁止使用 markdown 代码块包裹整段输出（不要输出 ``` 或 ```python）。
- 输出必须且只能包含 6 个变量的顺序赋值：project_json_src、host_tiling_src、host_operator_src、kernel_src、python_bind_src、model_src。
"""

ASCENDC_PITFALLS = """
AscendC 内核常见坑位（必须规避，否则编译失败或运行出错）：
- aicore 函数里禁止 float 与 unsigned 整型之间直接 `static_cast`，例如不要写
  `static_cast<float>(uint32_val)` 或 `static_cast<uint32_t>(float_val)`。
  需要互转时先绕一层 int32_t，例如：
    `static_cast<float>(static_cast<int32_t>(uintVal))`。
- tiling 结构体里如果某个 `uint32_t` 字段要参与浮点运算（例如作为归一化除数），
  建议在 host 侧预先算成 `float` 并放进独立字段，kernel 里直接读 float，避免
  在 aicore 函数里做整浮互转。
- `uint32_t` 循环计数器不要直接当浮点除数或乘数，也先转成 int32_t 再转 float。
- 所有从 tiling 传进来的标量，kernel 侧声明的类型必须与 tiling 结构体里的一致，
  否则会出现 "cast between floating and unsigned integer" 之类的错误。
"""

CONSTRAINT_SUFFIX = f"""

---
重要约束（必须严格遵守）：
1. 你只能单次直接生成完整答案，不能多次调用任何工具、检索或重新查询。
2. 你不能进行上板/编译测试，只能依赖自身的 LLM 能力和上下文进行推理。
3. 请将完整生成内容直接输出到 stdout，不要保存或写入任何文件。
{OUTPUT_FORMAT_RULES}
{ASCENDC_PITFALLS}
"""

CONSTRAINT_SUFFIX_WIKI = f"""

---
重要约束（必须严格遵守）：
1. 你可以使用 Read、Glob、Grep 工具查阅本地 AscendC Kernel Wiki（{WIKI_DIR}）中的参考资料，但不能调用其他任何工具。查阅完成后，必须一次性生成完整答案，不能多轮迭代修改。
2. 你不能进行上板/编译测试，只能依赖自身的 LLM 能力、上下文和 Wiki 参考信息进行推理。
3. 请将完整生成内容直接输出到 stdout，不要保存或写入任何文件。
{OUTPUT_FORMAT_RULES}
{ASCENDC_PITFALLS}
"""


EXPECTED_VARS = (
    'project_json_src',
    'host_tiling_src',
    'host_operator_src',
    'kernel_src',
    'python_bind_src',
    'model_src',
)

_VAR_ASSIGN_RE = re.compile(
    r'^(' + '|'.join(EXPECTED_VARS) + r')\s*=',
    re.MULTILINE,
)


def _extract_kernel_code(raw):
    """Strip preamble prose and markdown fences from Claude CLI output.

    Returns the cleaned source text, or None if the six required variable
    assignments aren't all present (likely a rate-limit / error response).
    """
    if not raw:
        return None
    text = raw.strip()

    # Unwrap a surrounding fenced code block, e.g. ```python\n...\n```
    fence_match = re.match(
        r'^```(?:python|py|cpp|c\+\+)?\s*\n(.*)\n```\s*$',
        text,
        re.DOTALL,
    )
    if fence_match:
        text = fence_match.group(1).strip()

    first_assign = _VAR_ASSIGN_RE.search(text)
    if first_assign is None:
        return None
    text = text[first_assign.start():]

    present = {m.group(1) for m in _VAR_ASSIGN_RE.finditer(text)}
    if not set(EXPECTED_VARS).issubset(present):
        return None

    return text.rstrip() + '\n'


def load_best_practices():
    with open(BEST_PRACTICES_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def build_prompt(language, strategy, op, best_practices=None, wiki=False):
    base = generate_prompt(language, strategy, op)
    parts = []
    if best_practices:
        parts.append("以下是 AscendC API 最佳实践参考，请在生成时严格遵循其中的规范：\n\n")
        parts.append(best_practices)
        parts.append("\n\n---\n\n")
    if wiki:
        parts.append(WIKI_PREAMBLE)
        parts.append("\n\n---\n\n")
    parts.append(base)
    parts.append(CONSTRAINT_SUFFIX_WIKI if wiki else CONSTRAINT_SUFFIX)
    return ''.join(parts)


def generate_with_claude_code(prompt, out_dir, op, timeout=300, disable_skills=False, with_wiki=False):
    """Invoke claude CLI in print mode."""
    out_path = os.path.join(out_dir, f'{op}.txt')
    if os.path.exists(out_path):
        print(f"[INFO] Already generated at {out_path}, skip")
        return

    claude_bin = shutil.which("claude") or "/usr/bin/claude"
    cmd = [
        claude_bin,
        "-p",
        "--output-format", "text",
    ]
    if with_wiki:
        cmd.extend(["--allowed-tools", "Read Glob Grep"])
    else:
        cmd.extend([
            "--disallowed-tools",
            "Read Glob Grep Write Edit Bash Task WebFetch WebSearch NotebookEdit",
        ])
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

    cleaned = _extract_kernel_code(out)
    if cleaned is None:
        raw_path = os.path.join(out_dir, f'{op}.raw.txt')
        with open(raw_path, 'w') as f:
            f.write(out)
        print(f"[SKIP] {op}: could not locate all six required variables; raw output saved to {raw_path}")
        return

    if cleaned != out:
        raw_path = os.path.join(out_dir, f'{op}.raw.txt')
        with open(raw_path, 'w') as f:
            f.write(out)
        print(f"[INFO] {op}: stripped {len(out) - len(cleaned)} bytes of preamble/fence; raw saved to {raw_path}")

    with open(out_path, 'w') as f:
        f.write(cleaned)


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
    parser.add_argument('--timeout', type=int, default=1200,
                        help='Timeout per operator in seconds')
    parser.add_argument('--disable-skills', action='store_true',
                        help='Pass --disable-slash-commands to claude CLI')
    parser.add_argument('--with-best-practices', action='store_true',
                        help='Prepend ascendc-api-best-practices.md to the prompt')
    parser.add_argument('--with-wiki', action='store_true',
                        help='Enable local wiki retrieval via Read/Glob/Grep tools')

    args = parser.parse_args()
    language = 'ascendc'
    ops = resolve_ops(args)

    best_practices = load_best_practices() if args.with_best_practices else None
    with_wiki = args.with_wiki

    print(f"Model name      : {args.model_name}")
    print(f"Strategy        : {args.strategy}")
    print(f"Categories      : {args.categories}")
    print(f"Ops (count={len(ops)}): {ops[:5]}{'...' if len(ops) > 5 else ''}")
    print(f"Disable skills  : {args.disable_skills}")
    print(f"With BP         : {args.with_best_practices}"
          + (f" ({len(best_practices)} chars)" if best_practices else ""))
    print(f"With Wiki       : {with_wiki}"
          + (f" ({WIKI_DIR})" if with_wiki else ""))

    for run in range(args.runs):
        out_dir = f'output/{language}/{args.strategy}/{temperature}-{top_p}/{args.model_name}/run{run}'
        os.makedirs(out_dir, exist_ok=True)

        for op in ops:
            try:
                prompt = build_prompt(language, args.strategy, op,
                                      best_practices=best_practices,
                                      wiki=with_wiki)
                generate_with_claude_code(prompt, out_dir, op,
                                          timeout=args.timeout,
                                          disable_skills=args.disable_skills,
                                          with_wiki=with_wiki)
            except Exception as e:
                print(f"[ERROR] Failed for {op}: {e}")


if __name__ == '__main__':
    main()
