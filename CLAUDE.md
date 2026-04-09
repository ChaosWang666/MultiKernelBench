# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MultiKernelBench is a benchmark for evaluating LLMs' ability to generate custom AscendC kernels for Ascend NPUs.

## Common Commands

### Setup
```bash
conda create --name multi-kernel-bench python=3.10
conda activate multi-kernel-bench
pip install -r requirements.txt
# NPU: pip install torch-npu==2.7.1
```

### Generate kernels via LLM
```bash
python generate_and_write.py --model deepseek-chat --strategy add_shot --categories activation
```
Arguments: `--runs`, `--model`, `--strategy`, `--categories` (use `all` for all categories).

### Evaluate generated kernels
```bash
python evaluation.py --model deepseek-chat --strategy add_shot --categories activation
```

### Generate baseline performance statistics
```bash
python generate_baseline_statistics.py
```

### API
Uses a unified API endpoint (api3.xhub.chat) with a pre-configured key. No environment variables needed. The `--model` argument selects the LLM (e.g., `deepseek-chat`, `qwen-plus`).

## Architecture

### Two-phase workflow
1. **Generation** (`generate_and_write.py`): Sends prompts to the LLM API, extracts generated kernel code, saves to `output/ascendc/{strategy}/{temperature}-{top_p}/{model}/run{run}/`
2. **Evaluation** (`evaluation.py` → `eval_single_runner.py` → `utils/evaluation_utils.py`): Compiles generated code, runs correctness checks (Model vs ModelNew with `torch.allclose`), measures performance, saves results as JSON.

### Plugin systems (registry pattern)
- **Backend** (`backends/ascendc_backend.py`): AscendC backend registered via `@register_backend('ascendc')`. Implements `get_device()`, `get_hardware_name()`, `compile()`, `correctness_execution()`, `time_execution()`, `cleanup()`.
- **Prompt strategies** (`prompt_generators/`): `ascendc_add_shot` and `ascendc_selected_shot`, registered via `@register_prompt("ascendc", strategy_name)`. Inherit from `BasePromptStrategy` and implement `generate(op)`.

### Reference implementations (`reference/`)
PyTorch reference implementations organized by category (activation, attention, convolution, matmul, etc.). Each file defines a `Model` class with `get_inputs()` and `get_init_inputs()` methods. The `--categories` argument maps to these directory names.

### Key modules
- `config.py`: Global settings — LLM parameters (temperature, top_p, max_tokens), trial counts, Ascend device config
- `dataset.py`: Task definitions mapping operations to categories (~900 lines)
- `utils/evaluation_utils.py`: Core `eval_single()` function — compilation, correctness, and performance measurement
- `utils/correctness.py` / `utils/performance.py`: Correctness testing (atol=1e-4, rtol=1e-4) and timing utilities
- `utils/utils.py`: Unified LLM client setup (OpenAI-compatible API via api3.xhub.chat), file I/O helpers

### Evaluation metrics
Results JSON contains: `compiled` (bool), `correctness` (bool), `performance` (mean/std/min/max/count), `hardware`, `errors`. Default: 5 correctness trials, 100 performance trials, 3 warmup runs.
