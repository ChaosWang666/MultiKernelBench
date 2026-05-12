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
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_and_write import generate_prompt
from dataset import dataset
from config import temperature, top_p
from generate_with_claude_code import generate_with_claude_code
from utils.claude_sdk_adapter import SUBMIT_TOOL_DIRECTIVE


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

BEST_PRACTICES_PATH = os.path.join(REPO_ROOT, 'ascendc-api-best-practices.md')
WIKI_DIR = os.path.join(REPO_ROOT, 'AscendC-Kernel-Wiki', 'wiki')

WIKI_PREAMBLE = f"""重要提示：
- Ascend C编程任务中，垂域知识缺失或困惑会导致失败（例如某个Ascend C API不能100%写出其签名等、编程范式和设计决策不确定、工具链使用问题、算子编译或执行报错不确定原因等）
，直接调用`cann-ask`skill查询相关知识以辅助问题高效解决。
- 至少调用一次 `cann-ask` skill。即使你认为自己会写任务算子，也必须先查一次高价值知识再写。
"""

OUTPUT_FORMAT_RULES = f"""
输出格式（最终交付必须严格遵守）：
{SUBMIT_TOOL_DIRECTIVE}
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
1. 若需AscendC领域参考资料，只能直接调用`cann-ask`查询，不能调用其他任何工具。查阅完成后，必须一次性生成完整答案，不能多轮迭代修改。
2. 对于通用领域问题（非Ascend C直接相关）、上下文中已知信息、确定清楚的信息，禁止使用`cann-ask`查询。禁止相似问题多次调用`cann-ask`查询。
3. 你不能进行上板/编译测试，只能依赖自身的 LLM 能力、上下文和 Wiki 参考信息进行推理。
4. 请将完整生成内容直接输出到 stdout，不要保存或写入任何文件。
{OUTPUT_FORMAT_RULES}
{ASCENDC_PITFALLS}
"""


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


WIKI_SKILL_TOOLS = ['Skill(cann-ask)', 'Skill(setup-cann-wiki)']


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
                generate_with_claude_code(
                    prompt, out_dir, op,
                    timeout=args.timeout,
                    disable_skills=args.disable_skills,
                    with_wiki=with_wiki,
                    extra_allowed_tools=WIKI_SKILL_TOOLS if with_wiki else None,
                )
            except Exception as e:
                print(f"[ERROR] Failed for {op}: {e}")


if __name__ == '__main__':
    main()
