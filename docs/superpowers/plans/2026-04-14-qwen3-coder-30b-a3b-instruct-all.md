# Qwen3 Coder 30B A3B Instruct Full Benchmark Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run MultiKernelBench end-to-end for `qwen3-coder-30b-a3b-instruct` across all categories, then summarize generation and evaluation results.

**Architecture:** Use the repository's existing two-phase workflow without changing source code. First verify the runtime environment and dataset scope, then execute `generate_and_write.py` for `--categories all`, then execute `evaluation.py` against the generated outputs, and finally summarize the resulting artifacts and failures.

**Tech Stack:** Python 3.10, MultiKernelBench CLI entrypoints, OpenAI-compatible model client, Ascend/NPU runtime, JSON result artifacts

---

## File Structure

- Read: `config.py` — confirms project root behavior, generation/evaluation parameters, and Ascend device target
- Read: `dataset.py` — confirms the benchmark op universe used by `--categories all`
- Read: `generate_and_write.py` — confirms generation command, output directory shape, and model-name path handling
- Read: `evaluation.py` — confirms evaluation command and final JSON output path
- Create: `docs/superpowers/specs/2026-04-14-qwen3-coder-30b-a3b-instruct-all-design.md` — approved design spec already written
- Create: `docs/superpowers/plans/2026-04-14-qwen3-coder-30b-a3b-instruct-all.md` — this execution plan
- Create: `output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0/` — generated model outputs and final benchmark results

### Task 1: Verify benchmark environment and scope

**Files:**
- Read: `config.py:3-25`
- Read: `dataset.py:1-140`
- Read: `requirements.txt:1-20`

- [ ] **Step 1: Confirm the configured benchmark parameters**

Inspect these settings in `config.py`:
```python
project_root_path = os.getcwd()
num_correct_trials = 5
num_perf_trials = 100
num_warmup = 3
temperature = 0.0
top_p = 1.0
ascendc_device = 'ai_core-Ascend910B2'
```
Expected: output paths will use `0.0-1.0`, and evaluation expects an Ascend 910B2-compatible environment.

- [ ] **Step 2: Confirm that `--categories all` maps to the full dataset**

Inspect the benchmark selection logic in `generate_and_write.py` and `evaluation.py`:
```python
op_tested = list(dataset.keys())
if categories != ['all']:
    op_tested = [op for op in op_tested if dataset[op]['category'] in categories]
```
Expected: with `--categories all`, every key in `dataset` is included.

- [ ] **Step 3: Count the benchmark operations before running**

Run:
```bash
python - <<'PY'
from dataset import dataset
print(len(dataset))
PY
```
Expected: prints a positive integer total op count used later in the summary.

- [ ] **Step 4: Verify the Python environment can import the project dependencies**

Run:
```bash
python - <<'PY'
import openai, torch, pytest
print('imports-ok')
PY
```
Expected: prints `imports-ok` with no ImportError.

- [ ] **Step 5: Verify the Ascend runtime prerequisite is visible**

Run:
```bash
python - <<'PY'
from config import ascendc_device
print(ascendc_device)
PY
```
Expected: prints `ai_core-Ascend910B2`. If the surrounding runtime is missing, stop before generation/evaluation and report the blocker.

### Task 2: Run full generation for all operators

**Files:**
- Read: `generate_and_write.py:120-155`
- Create: `output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0/*.txt`

- [ ] **Step 1: Confirm the exact generation command and output root**

Use the repository's entrypoint shape:
```python
for run in range(runs):
    out_dir = f'output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}'
    os.makedirs(out_dir, exist_ok=True)
    generate_and_write(out_dir, language, model, op_tested, strategy)
```
Expected output root:
```text
output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0/
```

- [ ] **Step 2: Execute full generation**

Run:
```bash
python generate_and_write.py --model qwen3-coder-30b-a3b-instruct --strategy add_shot --categories all
```
Expected: per-op log lines like `[INFO] Generate kernel for op ...`, and output `.txt` files created under the run directory.

- [ ] **Step 3: Verify that generation artifacts exist**

Run:
```bash
python - <<'PY'
import os
root = 'output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0'
print(os.path.isdir(root))
print(sum(1 for name in os.listdir(root) if name.endswith('.txt') and not name.endswith('_cot.txt')))
PY
```
Expected: first line `True`, second line a positive integer near the dataset op count.

- [ ] **Step 4: Spot-check for generation failures before evaluation**

Run:
```bash
python - <<'PY'
import os
from dataset import dataset
root = 'output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0'
missing = [op for op in dataset if not os.path.exists(os.path.join(root, f'{op}.txt'))]
print(f'missing={len(missing)}')
print(missing[:20])
PY
```
Expected: `missing=0`. If not zero, keep the list for the final summary because evaluation will fail for missing inputs.

### Task 3: Run full evaluation for all operators

**Files:**
- Read: `evaluation.py:8-59`
- Create: `output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0/result.json`

- [ ] **Step 1: Confirm the evaluation output path logic**

Inspect the output selection in `evaluation.py`:
```python
if categories == ['all']:
    output_file = os.path.join(out_dir,'result.json')
else:
    output_file = os.path.join(out_dir, f'result_{"_".join(categories)}.json')
```
Expected output file:
```text
output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0/result.json
```

- [ ] **Step 2: Execute full evaluation**

Run:
```bash
python evaluation.py --model qwen3-coder-30b-a3b-instruct --strategy add_shot --categories all
```
Expected: per-op log lines like `[INFO] Evaluating op ...` and a final JSON artifact unless the run stops on an environment-level failure.

- [ ] **Step 3: Verify the result artifact exists**

Run:
```bash
python - <<'PY'
import os
path = 'output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0/result.json'
print(os.path.exists(path))
PY
```
Expected: prints `True`.

- [ ] **Step 4: Stop immediately if the result file is missing**

If Step 3 prints `False`, report the blocking error from the evaluation command rather than retrying blindly. Common blockers to surface exactly as seen: model routing errors, missing generated files, or Ascend environment problems.

### Task 4: Summarize benchmark results

**Files:**
- Read: `output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0/result.json`

- [ ] **Step 1: Compute the top-level benchmark summary**

Run:
```bash
python - <<'PY'
import json
path = 'output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0/result.json'
with open(path) as f:
    data = json.load(f)
compiled = sum(1 for v in data.values() if v.get('compiled'))
correct = sum(1 for v in data.values() if v.get('correctness'))
perf = sum(1 for v in data.values() if v.get('performance'))
print({'total': len(data), 'compiled': compiled, 'correctness': correct, 'performance': perf})
PY
```
Expected: prints a dictionary with aggregate counts.

- [ ] **Step 2: Extract representative failures for reporting**

Run:
```bash
python - <<'PY'
import json
from collections import Counter
path = 'output/ascendc/add_shot/0.0-1.0/qwen3-coder-30b-a3b-instruct/run0/result.json'
with open(path) as f:
    data = json.load(f)
errors = []
for op, item in data.items():
    if not item.get('correctness'):
        msg = item.get('correctness_info') or item.get('errors') or 'unknown'
        errors.append(str(msg))
print(Counter(errors).most_common(10))
PY
```
Expected: prints the most common failure reasons for concise reporting.

- [ ] **Step 3: Return the final report to the user**

The final response must include:
```text
- Exact generation command used
- Exact evaluation command used
- Output directory path
- result.json path
- Total op count
- Generated file count
- Compiled count
- Correctness-pass count
- Performance-available count
- Main failure reasons or blocking issue
```
Expected: a concise operational summary with paths the user can inspect directly.

## Self-Review

- **Spec coverage:** The plan covers the approved spec's execution scope, fixed model name, `add_shot`, single run, `--categories all`, output locations, failure handling, and final reporting.
- **Placeholder scan:** No TODO/TBD placeholders remain. Each executable step includes the exact command or exact code/config snippet being verified.
- **Type consistency:** Output paths, model name, strategy, and result file names consistently use `qwen3-coder-30b-a3b-instruct`, `add_shot`, and `0.0-1.0` throughout.
