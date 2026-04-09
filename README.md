# MultiKernelBench

A benchmark for evaluating LLMs' ability to generate AscendC kernels for Ascend NPUs.

## Directory Structure

```text
MultiKernelBench/
├── ascend_op_projects/     # Ascend operator projects and extensions
├── backends/               # Backend implementation (ascendc)
├── prompt_generators/      # Prompt strategy implementations
├── prompts/                # Prompt templates and related resources
├── reference/              # PyTorch reference implementations used for correctness checks
│   ├── activation/
│   ├── attention/
│   ├── convolution/
│   ├── matmul/
│   └── ...
├── utils/                  # Utility modules
├── config.py               # Global runtime and model configuration
├── dataset.py              # Dataset loading and task organization
├── generate_and_write.py   # Generate kernels and write to output directory
├── generate_baseline_statistics.py  # Generate baseline statistics across tasks/categories
├── evaluation.py           # End-to-end evaluation entrypoint
└── eval_single_runner.py   # Single task/category evaluation runner
```

`reference/` is organized by `category`. The `--categories` argument should use these directory names (e.g., `activation`, `attention`, `convolution`, `matmul`, etc.).

## Quick start

### Set up
```bash
conda create --name multi-kernel-bench python=3.10
conda activate multi-kernel-bench
pip install -r requirements.txt

# For NPU users:
pip install torch-npu==2.7.1
```
You can rent NPU resources from online platforms such as [autodl](https://www.autodl.com/home).

### Config
Set configurations in config.py, including temperature and top_p for LLM. For AscendC evaluation, set `ascendc_device = ai_core-<soc_version>`.

### API
The project uses a unified API endpoint (api3.xhub.chat) with a pre-configured API key. No environment variables need to be set. The `--model` argument selects which LLM to use (e.g., `deepseek-chat`, `qwen-plus`, etc.) and the aggregation platform routes accordingly.

### Generate kernels using LLM and write them to output
```bash
python generate_and_write.py --model deepseek-chat --strategy add_shot --categories activation
```
Generated code is saved in ```output/ascendc/{strategy}/{temperature}-{top_p}/{model_name}/run{run}```.

#### Available Arguments

- `--runs`: Number of runs (default: `1`)
- `--model`: Model name (default: `deepseek-chat`)
- `--strategy`: Prompt strategy type (default: `add_shot`)
- `--categories`: Space-separated list of categories (default: `activation`)  
  Use `all` to include all categories.

### Evaluation
```bash
python evaluation.py --model deepseek-chat --strategy add_shot --categories activation
```
Evaluation result is saved in ```output/ascendc/{strategy}/{temperature}-{top_p}/{model_name}/run{run}/result_{category}.json```.

## Adding a Prompting Strategy

To add a custom prompting strategy, follow these steps:
1. **Create a Python file:**  
   Add a new file under `prompt_generators/` named as:  
   `prompt_generators/ascendc_{strategy_name}.py`  

2. **Create a New Strategy Class**

   - Inherit from `BasePromptStrategy`.
   - Implement the `generate(self, op)` method.

2. **Register the Strategy**

   Use the `@register_prompt("ascendc", strategy_name)` decorator with the desired strategy name.

## Credits

This project uses code from [KernelBench](https://github.com/ScalingIntelligence/KernelBench), licensed under the MIT License.
