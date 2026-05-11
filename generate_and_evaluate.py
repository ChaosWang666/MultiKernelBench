#!/usr/bin/env python3
"""Generate AscendC kernels with Claude Code CLI AND evaluate them concurrently.

Pipeline:
  - Main thread: walk (run, op) pairs and invoke Claude Code to write {op}.txt
    (sequential, identical to generate_with_claude_code.py).
  - Background eval thread: as soon as {op}.txt is written, dequeue and run
    eval_single_runner.py for it (subprocess), persist incremental result to
    {out_dir}/result[_<cats>].json atomically.

Compared to running generate_with_claude_code.py followed by evaluation.py
sequentially, this overlaps NPU evaluation of op_i with LLM generation of
op_{i+1}, saving roughly N * T_eval of wall-clock per batch.
"""

import os
import sys
import json
import queue
import argparse
import threading
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_with_claude_code import (
    build_prompt,
    generate_with_claude_code,
    load_best_practices,
    resolve_ops,
    WIKI_DIR,
)
from config import temperature, top_p


_SENTINEL = object()


def _classify_eval_result(op, kernel_path, eval_timeout):
    """Run eval_single_runner.py for a single kernel and classify its result.

    Mirrors the per-op subprocess + exception handling in evaluation.py:50-86.
    Returns (result_dict_or_None, fatal). When fatal is True the worker
    should stop processing further ops (misconfigured project_root_path).
    """
    if not os.path.exists(kernel_path):
        return None, False

    with open(kernel_path, 'r') as f:
        response_txt = f.read()

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tf_in, \
            tempfile.NamedTemporaryFile(mode='r', delete=True) as tf_out:
        tf_in.write(response_txt)
        tf_in.flush()
        try:
            subprocess.run(
                ['python3', 'eval_single_runner.py',
                 tf_in.name, op, tf_out.name],
                check=True,
                text=True,
                timeout=eval_timeout,
                capture_output=True,
            )
            return json.load(tf_out), False
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ''
            if 'FileNotFoundError' in stderr:
                print(f"[EVAL] FATAL FileNotFoundError on {op}: "
                      "possibly incorrect 'project_root_path' in config.py")
                return None, True
            if e.returncode == -11:
                return {
                    'compiled': True, 'correctness': False,
                    'performance': None,
                    'correctness_info': 'Segmentation fault',
                }, False
            return {
                'compiled': True, 'correctness': False,
                'performance': None,
                'correctness_info': 'Unknown fault',
            }, False
        except subprocess.TimeoutExpired:
            return {
                'compiled': True, 'correctness': False,
                'performance': None,
                'correctness_info': 'Timeout fault',
            }, False


def _persist_atomic(output_file, result):
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    tmp = output_file + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, output_file)


def _eval_worker(q, output_file, result, eval_timeout):
    """Background consumer: pull (op, kernel_path) tuples and evaluate them."""
    drained = False
    while True:
        item = q.get()
        if item is _SENTINEL:
            return
        if drained:
            continue
        op, kernel_path = item
        if op in result:
            print(f"[EVAL] {op} already in result.json, skip")
            continue
        if not os.path.exists(kernel_path):
            print(f"[EVAL] {op}: kernel file missing, skip silently")
            continue
        print(f"[EVAL] Starting {op}")
        item_result, fatal = _classify_eval_result(
            op, kernel_path, eval_timeout)
        if fatal:
            drained = True
            continue
        if item_result is None:
            continue
        result[op] = item_result
        _persist_atomic(output_file, result)
        print(f"[EVAL] {op} done: compiled={item_result.get('compiled')} "
              f"correctness={item_result.get('correctness')}")


def _resolve_output_file(out_dir, args):
    """Mirror evaluation.py:8-12 output naming.

    - If --ops is given (ops list arbitrary, may span categories), use result.json.
    - If --categories is 'all', use result.json.
    - Otherwise, result_{cat1}_{cat2}...json (categories in user-supplied order).
    """
    if args.ops:
        return os.path.join(out_dir, 'result.json')
    if args.categories == ['all']:
        return os.path.join(out_dir, 'result.json')
    return os.path.join(out_dir, f'result_{"_".join(args.categories)}.json')


def _load_existing_result(output_file):
    if not os.path.exists(output_file):
        return {}
    try:
        with open(output_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load existing {output_file} ({e}), starting fresh")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Generate AscendC kernels via Claude Code AND evaluate "
                    "them concurrently (pipeline overlap).")
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
                        help='Generate only, do not evaluate (equivalent to '
                             'generate_with_claude_code.py)')

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
            for op in ops:
                try:
                    prompt = build_prompt(
                        language, args.strategy, op,
                        best_practices=best_practices,
                        wiki=with_wiki,
                    )
                    generate_with_claude_code(
                        prompt, out_dir, op,
                        timeout=args.timeout,
                        disable_skills=args.disable_skills,
                        with_wiki=with_wiki,
                    )
                except Exception as e:
                    print(f"[ERROR] Generation failed for {op}: {e}")
                    continue

                if args.no_evaluate:
                    continue
                kernel_path = os.path.join(out_dir, f'{op}.txt')
                if os.path.exists(kernel_path):
                    q.put((op, kernel_path))
        finally:
            if eval_thread is not None:
                q.put(_SENTINEL)
                print("[INFO] Generation done, waiting for outstanding "
                      "evaluations...")
                eval_thread.join()
                print(f"[INFO] All evaluations done, results in {output_file}")


if __name__ == '__main__':
    main()
