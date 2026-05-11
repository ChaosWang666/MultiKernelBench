"""Parallel evaluator for MultiKernelBench.

Each worker has fully isolated resources (so they don't collide on shared
pybind install / opp deploy dir / CppExtension/csrc/op.cpp):

  - Independent venv at .worker_envs/w{i}/ (system-site-packages, only
    custom_ops_lib is installed locally)
  - Independent workspace ascend_op_projects_w{i}/ (CppExtension template
    is cloned from the canonical ascend_op_projects/CppExtension/)
  - Independent NPU device (worker_id % num_npus)

The original evaluation.py serial path is left untouched as a fallback.
"""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from config import temperature, top_p
from dataset import dataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKER_ENVS_DIR = os.path.join(PROJECT_ROOT, '.worker_envs')
TEMPLATE_OP_DIR = os.path.join(PROJECT_ROOT, 'ascend_op_projects')


@dataclass
class WorkerSpec:
    worker_id: int
    device_id: int
    op_engineer_dir: str   # ascend_op_projects_w{i}
    python_bin: str        # .worker_envs/w{i}/bin/python
    venv_bin_dir: str      # .worker_envs/w{i}/bin (prepended to PATH)


_WORKER_SPEC: WorkerSpec | None = None


def detect_num_npus() -> int:
    """Best-effort detection of available Ascend NPUs. Falls back to 1."""
    try:
        davinci = [f for f in os.listdir('/dev') if f.startswith('davinci') and f[7:].isdigit()]
        if davinci:
            return len(davinci)
    except OSError:
        pass
    return 1


def ensure_worker_envs(workers: int, num_npus: int) -> list[WorkerSpec]:
    """Idempotently set up per-worker venv + workspace. Returns specs."""
    os.makedirs(WORKER_ENVS_DIR, exist_ok=True)
    specs = []
    for i in range(workers):
        venv_dir = os.path.join(WORKER_ENVS_DIR, f'w{i}')
        venv_bin = os.path.join(venv_dir, 'bin')
        python_bin = os.path.join(venv_bin, 'python')
        workspace = os.path.join(PROJECT_ROOT, f'ascend_op_projects_w{i}')
        cpp_ext = os.path.join(workspace, 'CppExtension')

        if not os.path.exists(python_bin):
            print(f'[setup] creating venv {venv_dir} ...')
            subprocess.run(
                [sys.executable, '-m', 'venv', '--system-site-packages', venv_dir],
                check=True,
            )

        if not os.path.exists(cpp_ext):
            src = os.path.join(TEMPLATE_OP_DIR, 'CppExtension')
            if not os.path.exists(src):
                raise RuntimeError(
                    f'CppExtension template not found at {src}; '
                    f'run the serial pipeline once to populate it.'
                )
            print(f'[setup] cloning CppExtension -> {cpp_ext} ...')
            os.makedirs(workspace, exist_ok=True)
            # Skip build artifacts; stale wheels from prior Python versions
            # (e.g. cp313) would break `pip install *.whl` on the worker's
            # cp311 venv. Each worker rebuilds these on first op.
            shutil.copytree(
                src, cpp_ext,
                ignore=shutil.ignore_patterns('dist', 'build', '*.egg-info'),
            )

        specs.append(WorkerSpec(
            worker_id=i,
            device_id=i % num_npus,
            op_engineer_dir=workspace,
            python_bin=python_bin,
            venv_bin_dir=venv_bin,
        ))
    return specs


def _init_worker(id_queue) -> None:
    """ProcessPoolExecutor initializer: each pool process claims one WorkerSpec."""
    global _WORKER_SPEC
    _WORKER_SPEC = id_queue.get()


def _persist(output_file: str, result: dict) -> None:
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    tmp = output_file + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, output_file)


def _classify_result(op: str, response_txt: str, spec: WorkerSpec) -> tuple[str, dict, bool]:
    """Run one op evaluation in an isolated worker environment.

    Returns (op, result_item, fatal) where fatal=True signals the parent
    to abort the whole run (e.g., misconfigured project_root_path).
    """
    env = os.environ.copy()
    env['ASCEND_DEVICE_ID'] = str(spec.device_id)
    env['OP_ENGINEER_DIR'] = spec.op_engineer_dir
    # Put venv's bin first on PATH so build_and_run.sh's `python3`/`pip3`
    # resolve to the worker's venv, not the global interpreter.
    env['PATH'] = f"{spec.venv_bin_dir}:{env.get('PATH', '')}"

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tf_input, \
         tempfile.NamedTemporaryFile(mode='r', delete=True) as tf_output:
        tf_input.write(response_txt)
        tf_input.flush()
        try:
            subprocess.run(
                [spec.python_bin, 'eval_single_runner.py', tf_input.name, op, tf_output.name],
                check=True,
                text=True,
                timeout=180,
                env=env,
                cwd=PROJECT_ROOT,
                capture_output=True,
            )
            return op, json.load(tf_output), False
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ''
            if 'FileNotFoundError' in stderr:
                return op, {
                    'compiled': False, 'correctness': False, 'performance': None,
                    'correctness_info': "FileNotFoundError — check project_root_path"
                }, True
            if e.returncode == -11:
                return op, {
                    'compiled': True, 'correctness': False, 'performance': None,
                    'correctness_info': 'Segmentation fault'
                }, False
            return op, {
                'compiled': True, 'correctness': False, 'performance': None,
                'correctness_info': 'Unknown fault'
            }, False
        except subprocess.TimeoutExpired:
            return op, {
                'compiled': True, 'correctness': False, 'performance': None,
                'correctness_info': 'Timeout fault'
            }, False


def eval_one_op(op: str, out_dir: str) -> tuple[str, dict, bool]:
    """Pool task. Reads the generated kernel txt, dispatches to the runner."""
    assert _WORKER_SPEC is not None, 'worker initializer did not run'
    spec = _WORKER_SPEC
    op_txt_path = os.path.join(out_dir, f'{op}.txt')
    try:
        with open(op_txt_path, 'r') as f:
            response_txt = f.read()
    except FileNotFoundError:
        return op, {
            'compiled': False, 'correctness': False, 'performance': None,
            'correctness_info': f'Generated kernel file not found: {op_txt_path}'
        }, False
    return _classify_result(op, response_txt, spec)


def eval_all_parallel(out_dir: str, categories: list[str], op_tested, workers: int, num_npus: int) -> None:
    op_tested = list(op_tested)
    if categories == ['all']:
        output_file = os.path.join(out_dir, 'result.json')
    else:
        output_file = os.path.join(out_dir, f'result_{"_".join(categories)}.json')

    result = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                result = json.load(f)
        except Exception as e:
            print(f'[WARN] Failed to load existing {output_file} ({e}), starting fresh')
            result = {}

    pending = [op for op in op_tested if op not in result]
    if not pending:
        print(f'[INFO] All {len(op_tested)} ops already evaluated, see {output_file}')
        return
    if result:
        print(f'[INFO] Resuming: {len(result)} done, {len(pending)} remaining (out of {len(op_tested)})')

    specs = ensure_worker_envs(workers, num_npus)
    print(f'[INFO] Workers: {workers}, NPUs: {num_npus}')
    for s in specs:
        print(f'[INFO]   w{s.worker_id} -> npu:{s.device_id}, workspace={os.path.basename(s.op_engineer_dir)}')

    ctx = multiprocessing.get_context('spawn')
    manager = ctx.Manager()
    id_queue = manager.Queue()
    for spec in specs:
        id_queue.put(spec)

    fatal_seen = False
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(id_queue,),
    ) as pool:
        futures = {pool.submit(eval_one_op, op, out_dir): op for op in pending}
        for fut in as_completed(futures):
            op = futures[fut]
            try:
                op_done, result_item, fatal = fut.result()
            except Exception as e:
                print(f'[FAIL] op {op} task raised: {e}')
                result[op] = {
                    'compiled': False, 'correctness': False, 'performance': None,
                    'correctness_info': f'Task crashed: {e}'
                }
                _persist(output_file, result)
                continue
            result[op_done] = result_item
            _persist(output_file, result)
            print(f'[INFO] {op_done} -> {result_item}')
            if fatal:
                print('[FAIL] fatal error encountered, aborting remaining tasks')
                fatal_seen = True
                # cancel any not-yet-running futures; running ones will finish
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                break

    print(f'[INFO] Evaluated{"" if not fatal_seen else " (aborted)"}, written into {output_file}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parallel evaluator for MultiKernelBench.')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--model', type=str, default='deepseek-chat', help='Model name')
    parser.add_argument('--strategy', type=str, default='add_shot', help='Strategy type')
    parser.add_argument('--categories', nargs='+', default=['activation'], help='List of categories')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel worker processes (default 2). '
                             'Each worker has its own venv, workspace and NPU slot.')
    parser.add_argument('--npus', type=int, default=None,
                        help='Number of NPUs to use (default: auto-detect). '
                             'Worker i is bound to NPU (i %% npus).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_npus = args.npus if args.npus else detect_num_npus()
    if args.workers < 1:
        raise SystemExit('--workers must be >= 1')

    language = 'ascendc'
    print(f'Runs: {args.runs}')
    print(f'Model: {args.model}')
    print(f'Language: {language}')
    print(f'Strategy: {args.strategy}')
    print(f'Categories: {args.categories}')
    print(f'Workers: {args.workers}')
    print(f'NPUs:    {num_npus}')

    op_tested = list(dataset.keys())
    if args.categories != ['all']:
        op_tested = [op for op in op_tested if dataset[op]['category'] in args.categories]

    model_name = args.model.split('/', 1)[1] if '/' in args.model else args.model

    for run in range(args.runs):
        out_dir = f'output/{language}/{args.strategy}/{temperature}-{top_p}/{model_name}/run{run}'
        eval_all_parallel(out_dir, args.categories, op_tested, args.workers, num_npus)


if __name__ == '__main__':
    main()
