#!/usr/bin/env python3
"""Parallel Claude Code kernel generator with immediate-eval pipeline.

K-way parallel generation via asyncio.gather + Semaphore over the
claude-agent-sdk; each generated kernel is immediately handed to a single
background eval consumer thread (same pattern as generate_and_evaluate.py).

Why this layout:
  - The bottleneck is generation (~30-1200s per op via the SDK, which
    fork-execs the claude CLI under the hood); K-way parallel gen gives
    ~K x throughput on the gen phase.
  - Evaluation is NOT the bottleneck, so a single eval consumer thread is
    sufficient and avoids per-worker venv/NPU isolation work.
  - The eval consumer stays on a real OS thread (queue.Queue is
    thread-safe). The generation side moved to asyncio because per-task
    cost / num_turns / limit observability is async-native in the SDK.

K is configurable via --gen-workers (default 2) or env GEN_WORKERS.
Same --no-evaluate, resume, and result.json semantics as
generate_and_evaluate.py.
"""

import asyncio
import os
import sys
import queue
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_with_claude_code import (
    agenerate_with_claude_code,
    build_prompt,
    load_best_practices,
    resolve_ops,
    WIKI_DIR,
)
from generate_and_evaluate import (
    _SENTINEL,
    _eval_worker,
    _resolve_output_file,
    _load_existing_result,
)
from config import temperature, top_p


async def _gen_one_op(sem, language, strategy, op, best_practices, with_wiki,
                      out_dir, timeout, disable_skills, q, no_evaluate):
    """One coroutine per op. The semaphore caps in-flight SDK queries to K.
    On completion (success or skip), enqueue for the eval consumer if a
    {op}.txt was actually written; failures are swallowed so peers continue.
    """
    async with sem:
        try:
            prompt = build_prompt(language, strategy, op,
                                  best_practices=best_practices,
                                  wiki=with_wiki)
            await agenerate_with_claude_code(
                prompt, out_dir, op,
                timeout=timeout,
                disable_skills=disable_skills,
                with_wiki=with_wiki,
            )
        except Exception as e:
            print(f"[ERROR] Gen failed for {op}: {e}")
            return

    if no_evaluate:
        return
    kernel_path = os.path.join(out_dir, f'{op}.txt')
    if os.path.exists(kernel_path):
        q.put((op, kernel_path))


async def _drive_generation(ops, language, args, best_practices, with_wiki,
                            out_dir, q):
    sem = asyncio.Semaphore(args.gen_workers)
    tasks = [
        asyncio.create_task(_gen_one_op(
            sem, language, args.strategy, op,
            best_practices, with_wiki, out_dir,
            args.timeout, args.disable_skills,
            q, args.no_evaluate,
        ))
        for op in ops
    ]
    await asyncio.gather(*tasks, return_exceptions=False)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Claude Code kernel generator with immediate "
                    "eval (K-way gen + single eval consumer).")
    parser.add_argument('--model-name', type=str, default='claude-code',
                        help='Output directory model name')
    parser.add_argument('--strategy', type=str, default='add_shot')
    parser.add_argument('--categories', nargs='+', default=['fuse'],
                        help='Dataset categories to generate (default: fuse)')
    parser.add_argument('--ops', nargs='+', default=None,
                        help='Explicit operator list, overrides --categories')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=1200,
                        help='Generation timeout per operator (s)')
    parser.add_argument('--eval-timeout', type=int, default=180,
                        help='Evaluation timeout per operator (s)')
    parser.add_argument('--disable-skills', action='store_true',
                        help='Pass --disable-slash-commands to claude CLI')
    parser.add_argument('--with-best-practices', action='store_true',
                        help='Prepend ascendc-api-best-practices.md to prompt')
    parser.add_argument('--with-wiki', action='store_true',
                        help='Enable local AscendC Kernel Wiki retrieval')
    parser.add_argument('--no-evaluate', action='store_true',
                        help='Generate only, do not evaluate')
    parser.add_argument('--gen-workers', type=int,
                        default=int(os.environ.get('GEN_WORKERS', '2')),
                        help='Concurrent Claude CLI generations '
                             '(default 2, or $GEN_WORKERS)')

    args = parser.parse_args()
    if args.gen_workers < 1:
        raise SystemExit('--gen-workers must be >= 1')

    language = 'ascendc'
    ops = resolve_ops(args)

    best_practices = load_best_practices() if args.with_best_practices else None
    with_wiki = args.with_wiki

    print(f"Model name      : {args.model_name}")
    print(f"Strategy        : {args.strategy}")
    print(f"Categories      : {args.categories}")
    print(f"Ops (count={len(ops)}): {ops[:5]}{'...' if len(ops) > 5 else ''}")
    print(f"Gen workers     : {args.gen_workers}")
    print(f"Disable skills  : {args.disable_skills}")
    print(f"With BP         : {args.with_best_practices}"
          + (f" ({len(best_practices)} chars)" if best_practices else ""))
    print(f"With Wiki       : {with_wiki}"
          + (f" ({WIKI_DIR})" if with_wiki else ""))
    print(f"Evaluate        : {not args.no_evaluate} "
          f"(eval timeout {args.eval_timeout}s)")

    for run in range(args.runs):
        out_dir = (f'output/{language}/{args.strategy}/{temperature}-{top_p}/'
                   f'{args.model_name}/run{run}')
        os.makedirs(out_dir, exist_ok=True)

        output_file = _resolve_output_file(out_dir, args)
        result = _load_existing_result(output_file)
        if result:
            print(f"[INFO] Resuming with {len(result)} existing entries in "
                  f"{output_file}")

        q = queue.Queue()
        eval_thread = None
        if not args.no_evaluate:
            eval_thread = threading.Thread(
                target=_eval_worker,
                args=(q, output_file, result, args.eval_timeout),
                daemon=False,
            )
            eval_thread.start()

        try:
            asyncio.run(_drive_generation(
                ops, language, args, best_practices, with_wiki, out_dir, q,
            ))
        finally:
            if eval_thread is not None:
                q.put(_SENTINEL)
                print("[INFO] Generation done, waiting for outstanding "
                      "evaluations...")
                eval_thread.join()
                print(f"[INFO] All evaluations done, results in {output_file}")


if __name__ == '__main__':
    main()
