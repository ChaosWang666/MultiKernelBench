import os
import json
from dataset import dataset, category2exampleop
import subprocess
import tempfile
from config import temperature, top_p
import argparse
def eval_all(out_dir, categories, op_tested=dataset.keys()):
    if categories == ['all']:
        output_file = os.path.join(out_dir,'result.json')
    else:
        output_file = os.path.join(out_dir, f'result_{"_".join(categories)}.json')

    # 把 result.json 当增量 checkpoint 用：启动时读已有进度，
    # 每跑完一个算子立即写盘，中断后再跑只补未完成的算子。
    result = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                result = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load existing {output_file} ({e}), starting fresh")
            result = {}

    op_tested = list(op_tested)
    pending = [op for op in op_tested if op not in result]
    if not pending:
        print(f"[INFO] All {len(op_tested)} ops already evaluated, see {output_file}")
        return
    if len(result) > 0:
        print(f"[INFO] Resuming: {len(result)} done, {len(pending)} remaining (out of {len(op_tested)})")

    def _persist():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

    for op in pending:
        print(f"[INFO] Evaluating op {op}")
        op_txt_path = os.path.join(out_dir, f'{op}.txt')
        try:
            with open(op_txt_path, 'r') as saved_log:
                response_txt = saved_log.read()
        except FileNotFoundError:
            print(f"[FAIL] Generated kernel file not found: {op_txt_path}")
            result[op] = {'compiled': False, 'correctness': False, 'performance': None,
                          'correctness_info': f'Generated kernel file not found: {op_txt_path}'}
            _persist()
            continue
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tf_input, \
            tempfile.NamedTemporaryFile(mode='r', delete=True) as tf_output:

            tf_input.write(response_txt)
            tf_input.flush()
            try:
                subprocess.run(
                    ['python3', 'eval_single_runner.py', tf_input.name, op, tf_output.name],
                    check=True,
                    text=True,
                    timeout=180
                )
                result_item = json.load(tf_output)

            except subprocess.CalledProcessError as e:
                if 'FileNotFoundError' in e.stderr:
                    print("[FAIL] FileNotFoundError - Possibly due to incorrect 'project_root_path' setting in config.py")
                    _persist()
                    break
                elif e.returncode == -11:
                    print("[FAIL] Segmentation fault" )
                    seg_result = {'compiled': True, 'correctness': False, 'performance': None, 'correctness_info': 'Segmentation fault'}
                    result[op] = seg_result
                    _persist()
                    continue
                else:
                    print("[FAIL] unknown error, please report or fix bug")
                    unknown_result = {'compiled': True, 'correctness': False, 'performance': None, 'correctness_info': 'Unknown fault'}
                    result[op] = unknown_result
                    _persist()
                    continue
            except subprocess.TimeoutExpired as e:
                print("[FAIL] run timeout")
                time_result = {'compiled': True, 'correctness': False, 'performance': None, 'correctness_info': 'Timeout fault'}
                result[op] = time_result
                _persist()
                continue
            result[op] = result_item
            _persist()
            print(f'[INFO] {result_item}')

    print(f"[INFO] Evaluated successfully, written into {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--model', type=str, default='deepseek-chat', help='Model name')
    parser.add_argument('--strategy', type=str, default='add_shot', help='Strategy type.')
    parser.add_argument('--categories', nargs='+', default=['activation'], help='List of categories.')

    args = parser.parse_args()

    runs = args.runs
    model = args.model
    language = 'ascendc'
    strategy = args.strategy
    categories = args.categories

    print(f"Runs: {runs}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Strategy: {strategy}")
    print(f"Categories: {categories}")

    op_tested = list(dataset.keys())
    if categories != ['all']:
        op_tested = [op for op in op_tested if dataset[op]['category'] in categories]

    if '/' in model:
        # handle slashed model names (e.g., provider/model-name)
        model_name = model.split('/')[1]
    else:
        model_name = model

    for run in range(runs):
        out_dir = f'output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}'
        eval_all(out_dir, categories, op_tested)
