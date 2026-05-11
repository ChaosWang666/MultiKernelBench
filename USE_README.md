# MultiKernelBench 使用手册

完整描述与**生成**和**评测**相关的所有源码文件的实际用法、代码结构和调用逻辑。

---

## 0. 全部文件总览

按"调用层级 → 文件 → 角色"组织。所有 `output/`、`baselines/` 路径都相对 repo 根目录。

| 层级 | 文件 | 角色 |
|---|---|---|
| **入口 · 仅生成** | `generate_and_write.py` | SDK 生成（OpenAI 兼容 + Google GenAI） |
|  | `generate_with_claude_code.py` | Claude CLI 生成（Read/Glob/Grep 工具检索 wiki） |
|  | `generate_with_cc_and_wiki.py` | Claude CLI 生成（`cann-ask` skill 检索 wiki） |
| **入口 · 生成+评测** | `generate_and_evaluate.py` | Claude CLI 顺序生成 + 单线程评测，1 gen + 1 eval 重叠 |
|  | `generate_and_evaluate_parallel.py` | K 路并行生成 + 单线程评测 |
| **入口 · 仅评测** | `evaluation.py` | 串行评测，启动时读 `result.json` 当 checkpoint |
|  | `evaluation_parallel.py` | W workers，每 worker 独立 venv + 工作目录 + NPU |
|  | `eval_single_runner.py` | 单算子评测子进程入口（17 行 thin wrapper） |
| **入口 · 基线** | `generate_baseline_statistics.py` | 测量参考实现 `Model` 的硬件基线性能 |
| **评测核心** | `utils/evaluation_utils.py::eval_single` | 五档流水：抽取代码 → 编译 → 正确性 → 性能 |
|  | `backends/ascendc_backend.py::AscendBackend` | 实现 Backend 接口的五个方法 |
|  | `backends/backend_registry.py` | `BACKEND_REGISTRY` + `@register_backend` 装饰器 |
|  | `utils/ascend_compile_pipeline.py::ascend_compile` | msopgen → build.sh → deploy → pybind 四阶段编译 |
|  | `utils/correctness.py::execute_template` | Model vs ModelNew，5 trial `torch.allclose(atol=1e-4)` |
|  | `utils/performance.py::time_execution_event_template` | 3 warmup + 100 trial，`torch_npu.npu.Event` 计时 |
| **生成核心** | `prompt_generators/prompt_registry.py` | `PROMPT_REGISTRY` + `@register_prompt` |
|  | `prompt_generators/ascendc_add_shot.py` | 始终用 `add` 算子作 few-shot 示例 |
|  | `prompt_generators/ascendc_selected_shot.py` | 按 `category2exampleop` 选示例算子 |
|  | `prompt_generators/prompt_utils.py` | prompt 模板与示例文件读取 |
| **基础设施** | `dataset.py` | 算子元数据字典 `dataset` + `category2exampleop` |
|  | `config.py` | 全局参数（trial 数、设备、路径） |
|  | `utils/utils.py` | OpenAI/Google client 工厂、`get_ref_src_path`、字符串工具 |

---

## 1. 共享基础设施

### 1.1 输出目录约定

```
output/
└── ascendc/
    └── {strategy}/                       # e.g. add_shot, selected_shot
        └── {temperature}-{top_p}/        # e.g. 0.0-1.0 （来自 config.py）
            └── {model_name}/             # e.g. claude-code, deepseek-chat
                └── run{N}/
                    ├── {op}.txt           # 生成的内核源码（六变量字符串）
                    ├── {op}_cot.txt       # 思维链（仅 SDK 路径，模型回了 reasoning_content）
                    ├── {op}.raw.txt       # 原始 LLM 输出（仅 Claude CLI 路径剥除前言后保留）
                    └── result[_<cats>].json   # 评测结果（增量 checkpoint）
```

`result.json` 命名：单类别 `all` 或显式 `--ops` 时是 `result.json`，否则 `result_<cat1>_<cat2>.json`。

基线性能落盘到 `baselines/{language}_{device}.json`（如 `baselines/ascendc_ai_core-Ascend910B2.json`）。

### 1.2 数据集与参考实现

- `dataset.py`：`dataset = {op_name: {category, ...}}`，配合 `category2exampleop = {category: example_op}`。
- `reference/{category}/{op}.py`：每个算子的 PyTorch 参考实现，必须包含 `Model` 类、`get_inputs()`、`get_init_inputs()` 三个符号。评测时 `exec(ref_src, context)` 把这三者注入运行环境。
- `prompts/cuda_model_{example}.py` + `prompts/{language}_new_model_{example}.py`：prompt 里 few-shot 用的"示例算子原始版"与"AscendC 改写版"。
- `utils.utils.get_ref_src_path(op)`：从 `dataset[op]['category']` 定位上面这个文件。

### 1.3 插件注册表（懒加载）

| 注册表 | 装饰器 | 当前实现 |
|---|---|---|
| `BACKEND_REGISTRY` | `@register_backend('ascendc')` 于 `backends/ascendc_backend.py` | `AscendBackend` |
| `PROMPT_REGISTRY` | `@register_prompt('ascendc', strategy)` 于 `prompt_generators/ascendc_*.py` | `add_shot`, `selected_shot` |

懒加载规则：首次访问时按 `language` 拼出模块名 `backends.{language}_backend` 或 `prompt_generators.{language}_{strategy}`，`importlib.import_module` 触发其中的装饰器把对象塞进注册表。

### 1.4 关键配置项（`config.py`）

```python
project_root_path = os.getcwd()
ref_impl_base_path = f'{project_root_path}/reference'

# trial 数（evaluation 与 baseline 都用）
max_turn = 1
num_correct_trials = 5
num_perf_trials = 100
num_warmup = 3

# LLM 采样
max_tokens = 8192
temperature = 0.0
top_p = 1.0
num_completions = 1
seed_num = 1024

# AscendC 编译
op_engineer_dir = os.environ.get('OP_ENGINEER_DIR', f'{project_root_path}/ascend_op_projects')
deploy_path = f'{op_engineer_dir}/opp'
ascendc_device = 'ai_core-Ascend910B2'   # 上报到 result.json 的 hardware 字段
```

环境变量覆盖点：`OP_ENGINEER_DIR`（并行评测每 worker 用 `ascend_op_projects_w{i}`）、`ASCEND_DEVICE_ID`（绑 NPU id）、`GEN_WORKERS`（并行生成默认 K）。

### 1.5 LLM API 客户端（`utils/utils.py`）

- `get_client(model)`：返回 OpenAI 兼容 client。注意：文件里有**两个同名定义**，第二个（阿里云 dashscope）会覆盖第一个（api3.xhub.chat），实际生效的是阿里云端点。API key 当前硬编码在源码里。
- `get_google_client()`：构造 `google.genai.Client`，挂在 `127.0.0.1:7899` 代理上，API key 同样硬编码。
- `get_ref_src_path(op)`：拼出 `reference/{category}/{op}.py`。
- `read_file(path)`：缺失/异常返回空串。
- `underscore_to_pascalcase(s)`：`vector_add → VectorAdd`，用于 AscendC 算子工程的命名约定。
- `extract_first_code(text, types)`：从 ` ``` `…` ``` ` 提取第一个代码块，注意它的 `code` 变量在没有匹配 `code_type` 时未定义（潜在 bug）。

---

## 2. 生成环节详解

### 2.0 Prompt 构造（`prompt_generators/`）

`build_prompt` / `generate_prompt` 的内部调用：

```
generate_prompt(language='ascendc', strategy='add_shot' or 'selected_shot', op)
 └─ PROMPT_REGISTRY['ascendc'][strategy].generate(op)
     └─ read_relavant_files('ascendc', op, example_op)
         ├─ prompts/cuda_model_{example_op}.py            ← 示例原始 PyTorch 模型
         ├─ prompts/ascendc_new_model_{example_op}.py     ← 示例 AscendC 改写版
         └─ reference/{category}/{op}.py                  ← 待替换的目标算子
     └─ ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, example_op)
         ├─ 把 op 名加后缀 _custom，PascalCase 化得到工程类名
         ├─ 拼上 ASCENDC_PROBLEM_STATEMENT
         ├─ 嵌入示例原版 + 示例 AscendC 版（few-shot）
         ├─ 嵌入目标算子源码
         └─ 拼上 ASCENDC_PROBLEM_INSTRUCTION
            （要求输出六个字符串：project_json_src, host_tiling_src,
             host_operator_src, kernel_src, python_bind_src, model_src）
```

两种策略的差异只在选哪个 `example_op`：

| 策略 | 示例算子 | 适用场景 |
|---|---|---|
| `add_shot` | 固定 `'add'`（最简单的逐元素算子） | 测试模型迁移到任意算子的泛化能力 |
| `selected_shot` | `category2exampleop[category]` | 给同类别算子提供更贴近的示例（如 normalization 类用 `batch_norm`） |

Claude CLI 路径在这个 base prompt 之上还会追加 `CONSTRAINT_SUFFIX`（禁用工具、输出格式约束、AscendC 常见坑位提示）；详见 2.2/2.3。

### 2.1 路径 A：SDK 生成（`generate_and_write.py`）

适用于 `deepseek-chat`、`qwen-plus`、`gemma-4-31b-it` 等通过统一 API 或 Google GenAI SDK 访问的模型。

```
main()
 └─ for run in runs:
     └─ generate_and_write(out_dir, language, model, op_tested, strategy)
         ├─ client = get_google_client() if model in GOOGLE_MODELS
         │           else get_client(model)
         └─ for op in op_tested:
             ├─ 若 {op}.txt 已存在 → skip（幂等）
             ├─ prompt = generate_prompt(language, strategy, op)
             └─ generate_and_write_single[_google](prompt, client, out_dir, op, model)
                 ├─ OpenAI: client.chat.completions.create(stream=True, ...)
                 │  ├─ 区分 delta.reasoning_content（思维链）与 delta.content（正式回答）
                 │  ├─ 写 {op}_cot.txt（若 reasoning_content 非空）
                 │  └─ 写 {op}.txt（answer_content）
                 └─ Google: client.models.generate_content_stream(...)
                     └─ 写 {op}.txt（只有最终回答）
```

**重试**：OpenAI 路径 `max_retries=3`，等待 5/10/20s 指数退避；Google 路径 `max_retries=5`，10/20/40/80/160s。

**CLI 用法**：

```bash
python generate_and_write.py \
    --model deepseek-chat \
    --strategy add_shot \
    --categories activation
```

参数：`--runs`（默认 1）、`--model`、`--strategy`（默认 `add_shot`）、`--categories`（默认 `['activation']`，`all` 表示全量）。

### 2.2 路径 B：Claude CLI（`generate_with_claude_code.py`）

用 `claude` 命令行单轮生成（`-p --output-format text`），通过 stdin 喂 prompt，从 stdout 读完整答复，再用正则剥除 markdown / 前言。

```
build_prompt(language, strategy, op, best_practices=None, wiki=False)
 ├─ base = generate_prompt(language, strategy, op)
 ├─ if best_practices: 前面拼接 ascendc-api-best-practices.md
 ├─ if wiki: 前面拼接 WIKI_PREAMBLE（介绍本地 wiki 目录与查询协议）
 └─ 后面拼接 CONSTRAINT_SUFFIX 或 CONSTRAINT_SUFFIX_WIKI
     ├─ 禁止前言/后记/markdown 围栏
     ├─ 要求第一个非空字符就是 `project_json_src` 的 `p`
     └─ ASCENDC_PITFALLS（float ↔ unsigned 互转、tiling 类型一致性）

generate_with_claude_code(prompt, out_dir, op, timeout, disable_skills, with_wiki)
 ├─ 若 {op}.txt 已存在 → skip
 ├─ 构造命令：
 │   claude -p --output-format text
 │     [--allowed-tools "Read Glob Grep" if with_wiki
 │      else --disallowed-tools "Read Glob Grep Write Edit Bash Task ..."]
 │     [--disable-slash-commands if disable_skills]
 ├─ subprocess.run(input=prompt, timeout=timeout, cwd=REPO_ROOT)
 ├─ rate-limit 检测：输出 < 200 字节 或含 "You've hit your limit" 等标志
 │   → 静默跳过（不落盘任何文件）
 ├─ _extract_kernel_code(out):
 │   ├─ 用正则剥除 ```python``` ```cpp``` 等围栏
 │   ├─ 从第一个 project_json_src= 起截断
 │   └─ 校验六变量是否齐全
 │   失败 → 写 {op}.raw.txt，返回
 ├─ 若清理后内容与原始不同 → 也写 {op}.raw.txt 供事后审计
 └─ 写 {op}.txt
```

三种 prompt 变体（同一个生成函数）：

| 变体 | claude CLI 参数 | prompt 前缀 |
|---|---|---|
| `baseline` | `--disallowed-tools "Read Glob Grep Write ..."` | 仅 base + CONSTRAINT_SUFFIX |
| `bp`       | 同上 | base + `ascendc-api-best-practices.md` + CONSTRAINT_SUFFIX |
| `wiki`     | `--allowed-tools "Read Glob Grep"` | base + WIKI_PREAMBLE + CONSTRAINT_SUFFIX_WIKI |

注意：本文件**不直接作为 CLI 入口使用**。它的 `main()` 还在，但实战中由 `generate_and_evaluate.py` / `generate_and_evaluate_parallel.py` 当库函数调用 `build_prompt` + `generate_with_claude_code`。

### 2.3 路径 B'：Claude CLI + cann-ask（`generate_with_cc_and_wiki.py`）

`generate_with_claude_code.py` 的演进版本：用 `Skill(cann-ask)` 替代 `Read/Glob/Grep`，把"本地 wiki 文件检索"升级为"Claude Code 内置 skill 调用"。

差异（与 2.2 路径 B 对比）：

```diff
- cmd.extend(["--allowed-tools", "Read Glob Grep"])
+ cmd.extend(["--allowed-tools", "Skill(cann-ask) Skill(setup-cann-wiki) Read Glob Grep"])

  WIKI_PREAMBLE: 不再描述目录结构，而是要求至少调用一次 cann-ask 查询高价值知识

  CONSTRAINT_SUFFIX_WIKI: 强调"只能使用 cann-ask 查询，禁止其它工具，
                          相似问题不要重复查询"
```

接口与参数完全对齐 `generate_with_claude_code.py`。同样不直接当 CLI 入口跑，按需 import 即可。

### 2.4 三个 Claude CLI 入口文件的实际选用

| 想测什么 | 选哪个 |
|---|---|
| 验证模型纯能力 | path B baseline（`--disable-skills`，不带 wiki/bp） |
| 加入最佳实践提示能否提升 | path B bp（`--with-best-practices`） |
| 让模型自己查本地 wiki | path B wiki（`--with-wiki`） |
| 升级到 cann-ask skill 检索 | path B'（用 `generate_with_cc_and_wiki.py` 替换 import） |

三者底层都是同一个 `claude` CLI 子进程 + 同样的六变量提取，差别只在 prompt 与允许的工具集合。

---

## 3. 评测环节详解

评测路径的核心实现只有一个：`utils/evaluation_utils.py::eval_single`。其它所有评测脚本（`evaluation.py` / `evaluation_parallel.py` / 两个 `generate_and_evaluate*`）都通过 `eval_single_runner.py` 子进程间接调用它。

### 3.1 评测核心：`eval_single(response_txt, op, language)`

按下面五档顺序往后走，任何一档失败立即返回，后续档不跑：

```
eval_single(response_txt, op, 'ascendc')
 ├─ 1) 准备 backend = BACKEND_REGISTRY[language]    # 懒加载 ascendc_backend
 │     hardware = backend.get_hardware_name()       # 'ai_core-Ascend910B2'
 │     result = {'compiled': False, 'correctness': False, 'performance': None, 'hardware': ...}
 ├─ 2) extract_first_code(response_txt, ['python', 'cpp'])
 │     若没找到 ``` 围栏 → 直接当成纯代码用
 ├─ 3) backend.compile(generated_code, op)          # 见 §3.4
 │     失败：result['compile_info'] = str(e); return
 │     成功：result['compiled'] = True
 ├─ 4) backend.correctness_execution(ref_src)       # 见 §3.5
 │     失败：result['correctness_info'] = info; return
 │     成功：result['correctness'] = True
 ├─ 5) backend.time_execution(eval_target='ModelNew')   # 见 §3.6
 │     elapsed_times: list[float] (毫秒)
 │     result['performance'] = {mean, std, min, max, num_trials}
 └─ backend.cleanup()                               # 释放 context, npu cache
```

返回的 `result` 字典 schema：

```json
{
  "compiled": true,
  "correctness": true,
  "performance": {
    "mean": 0.0123,
    "std": 0.0008,
    "min": 0.0119,
    "max": 0.0145,
    "num_trials": 100
  },
  "hardware": "ai_core-Ascend910B2"
}
```

或失败态：

```json
{
  "compiled": false,
  "correctness": false,
  "performance": null,
  "hardware": "ai_core-Ascend910B2",
  "compile_info": "Exit Code: 1\n[STDOUT]...[STDERR]gcc: error: ..."
}
```

### 3.2 子进程入口：`eval_single_runner.py`

```python
# 17 行 thin wrapper
response_txt_path = sys.argv[1]
op = sys.argv[2]
result_path = sys.argv[3]
response_txt = open(response_txt_path).read()
result = eval_single(response_txt, op, 'ascendc')
json.dump(result, open(result_path, 'w'))
```

为什么所有评测路径都通过子进程调用？

1. AscendC 编译会改 cwd、动态加载 `.so`、污染 `sys.modules`。子进程退出自动清理。
2. 段错误（SIGSEGV，returncode = -11）会被父进程的 `subprocess.CalledProcessError` 捕获，能继续评测下一个算子。
3. `timeout=180`（`config.num_perf_trials` 间接限制）能强行回收挂死的子进程。
4. 并行评测时每个 worker 用自己的 venv 调起这个子进程，互不干扰。

### 3.3 Backend 接口（`backends/backend_registry.py` + `backends/ascendc_backend.py`）

**抽象基类** `Backend`（`backends/backend_registry.py`）只声明六个方法：

```python
class Backend:
    def get_device(self): ...           # 返回 torch.device
    def get_hardware_name(self): ...    # 字符串，写进 result.json
    def compile(self, generated_code, op): ...      # → (bool, error_str)
    def correctness_execution(self, ref_src): ...   # → (bool, info_str)
    def time_execution(self, eval_target='ModelNew'): ...  # → list[float] (ms)
    def cleanup(self): ...
```

**AscendC 实现** `AscendBackend`（`backends/ascendc_backend.py`）：

```python
@register_backend('ascendc')
class AscendBackend(Backend):
    def __init__(self):
        self.context = {}                       # 跨 compile/correctness/timing 复用
        self.device = self.get_device()         # torch.device('npu:$ASCEND_DEVICE_ID')

    def get_hardware_name(self):
        return ascendc_device                   # config 里硬编码，避免 npu.get_device_name() 崩溃

    def compile(self, generated_code, op):
        try:
            ascend_compile(generated_code, op, self.context)   # 见 §3.4
            return True, None
        except Exception as e:
            os.chdir(project_root_path)         # 安全 reset cwd
            return False, str(e)

    def correctness_execution(self, ref_src):
        exec(ref_src, self.context)             # 注入 Model / ModelNew / get_inputs / get_init_inputs
        return execute_template(torch_npu.npu.synchronize, self.device, self.context)

    def time_execution(self, eval_target='ModelNew'):
        return time_execution_event_template(
            self.context, self.device,
            torch_npu.npu.synchronize, torch_npu.npu.Event, eval_target)

    def cleanup(self):
        del self.context
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize(device=self.device)
```

`context` 字典在 §3.4 编译完成后会包含六变量源（`project_json_src` 等）加上 `Model` / `ModelNew` / `get_inputs` / `get_init_inputs`，供 §3.5、§3.6 使用。

### 3.4 编译四阶段：`utils/ascend_compile_pipeline.py::ascend_compile`

把生成的"六变量源串"展开成完整 AscendC 工程，编译，部署 opp 包，并通过 pybind 安装到当前 venv。

```
ascend_compile(generated_code, op, context, extra_kernel_include_paths=None):
 ├─ op = op + '_custom'                                # 防止与官方算子冲突
 ├─ op_capital = underscore_to_pascalcase(op)          # 工程目录名
 ├─ target_directory = $op_engineer_dir/{op_capital}
 │
 ├─ 0) exec(generated_code, context)                   # 把六变量塞进 context
 │     语法错或 KeyError 直接 raise
 │
 ├─ 1) [msopgen] 创建算子工程
 │     若 target_directory 已存在 → shutil.rmtree
 │     写 {op}.json → subprocess.run(['msopgen', 'gen', '-i', ..., '-c', ascendc_device, '-lan', 'cpp', '-out', op_capital])
 │     失败 → raise Exception("Exit Code: ...\n[STDOUT]...[STDERR]...")
 │
 ├─ 2) [写入六个源文件] 到 msopgen 生成的工程结构里
 │     ├─ op_host/{op}_tiling.h         ← host_tiling_src
 │     ├─ op_host/{op}.cpp              ← host_operator_src
 │     ├─ op_kernel/{op}.cpp            ← kernel_src
 │     └─ ../CppExtension/csrc/op.cpp   ← python_bind_src
 │     若 extra_kernel_include_paths 非空 → 注入 op_kernel/CMakeLists.txt
 │
 ├─ 3) [build.sh] 编译生成 .so + custom_opp_*.run
 │     cwd → target_directory，subprocess.run(['./build.sh'])
 │     失败 → raise Exception(...)
 │
 ├─ 4) [deploy] 把 .run 安装到 OPP 路径
 │     cwd → target_directory/build_out
 │     glob "custom_opp_*.run" → subprocess.run([f'./{run_file}'])
 │     失败 → raise Exception(...)
 │
 ├─ 5) [pybind] 用 CppExtension 把算子绑给 Python
 │     cwd → $op_engineer_dir/CppExtension
 │     subprocess.run(['bash', './build_and_run.sh'])
 │     失败 → raise Exception(...)
 │
 ├─ 6) 设置环境变量（每个 worker 隔离）
 │     ASCEND_CUSTOM_OPP_PATH = {deploy_path}/vendors/customize
 │     LD_LIBRARY_PATH 前置 {deploy_path}/vendors/customize/op_api/lib/
 │
 ├─ 7) exec(context['model_src'], context)             # 实例化 ModelNew，要求 custom_ops_lib 能 import
 │
 └─ os.chdir(project_root_path)                        # 恢复 cwd
```

任何阶段失败都 raise，被 `AscendBackend.compile` 捕获后返回 `(False, str(e))`。整个流程严重依赖 cwd 切换，所以一次只能跑一个（并行评测必须用独立 `op_engineer_dir`，见 §3.8）。

### 3.5 正确性比对：`utils/correctness.py::execute_template`

```
execute_template(synchronize, device, context):
 ├─ 从 context 取出 get_inputs / get_init_inputs / Model / ModelNew
 ├─ init_inputs = get_init_inputs()  → 搬到 device
 ├─ 在 torch.no_grad() + set_seed(seed_num=1024) 下：
 │   ├─ original_model = Model(*init_inputs).to(device)
 │   └─ custom_model   = ModelNew(*init_inputs).to(device)
 │      （两次重设 seed 保证权重一致）
 └─ for trial in num_correct_trials=5:
     ├─ inputs = get_inputs() → 搬到 device
     ├─ synchronize 后跑 ref_output = Model(*inputs)
     ├─ synchronize 后跑 new_output = ModelNew(*inputs)
     ├─ synchronize
     └─ 校验：
         ├─ 形状必须一致
         └─ torch.allclose(ref, new, atol=1e-4, rtol=1e-4)
         任一失败 → return (False, "[FAIL] ...")
```

任何运行期异常 → `return (False, "[FAIL] {str(e)}")`。

### 3.6 性能测量：`utils/performance.py::time_execution_event_template`

```
time_execution_event_template(context, device, synchronize, event_class, eval_target='ModelNew'):
 ├─ inputs / init_inputs 各取一次（这次不 reseed），搬到 device
 ├─ custom_model = ModelNew(*init_inputs).to(device)   # 或 Model（baseline 用 'Model'）
 ├─ warmup: for _ in num_warmup=3:
 │   custom_model(*inputs); synchronize
 └─ for trial in num_perf_trials=100:
     start_event = event_class(enable_timing=True)
     end_event   = event_class(enable_timing=True)
     start_event.record()
     custom_model(*inputs)
     end_event.record()
     synchronize
     elapsed_times.append(start.elapsed_time(end))     # 毫秒
 return elapsed_times
```

不在循环里重建模型，所以测的是稳态执行时间。

### 3.7 三种调度策略

#### 策略 1：串行（`evaluation.py`）

```bash
python evaluation.py --model claude-code --strategy add_shot --categories activation
```

```
eval_all(out_dir, categories, op_tested):
 ├─ 解析 output_file = out_dir/result[_<cats>].json
 ├─ result = load existing checkpoint（容错：JSON 错误就重来）
 ├─ pending = [op for op in op_tested if op not in result]
 └─ for op in pending:
     ├─ 读 out_dir/{op}.txt
     │   FileNotFoundError → result[op] = stub('Generated kernel file not found'); persist; continue
     ├─ NamedTemporaryFile 写 kernel 内容
     ├─ subprocess.run(['python3', 'eval_single_runner.py', tf_in, op, tf_out], timeout=180)
     │   ├─ CalledProcessError + 'FileNotFoundError' in stderr → 致命错误，break
     │   ├─ returncode == -11 → stub('Segmentation fault'); persist; continue
     │   ├─ 其它 CalledProcessError → stub('Unknown fault'); persist; continue
     │   └─ TimeoutExpired → stub('Timeout fault'); persist; continue
     └─ result[op] = json.load(tf_out); persist
```

每跑完一个算子立刻 persist（atomic 通过 `_persist` 直接 dump）→ 中断后下次重跑只补 pending。

#### 策略 2：生成+评测重叠（`generate_and_evaluate.py`）

主线程顺序生成 Claude CLI，单个后台线程消费评测队列：

```
main thread                          background eval thread
-----------                          ----------------------
for op in ops:
  generate_with_claude_code(op) ──► writes {op}.txt
  if {op}.txt exists:
      queue.put((op, kernel_path)) ──► queue.get()
                                      subprocess.run('eval_single_runner.py' ...)
                                      classify result, atomic write result.json
# end of all gen
queue.put(SENTINEL)
eval_thread.join()
```

- `_persist_atomic`：tempfile + `os.replace`，Ctrl-C 也保证 `result.json` 合法。
- 致命错误（`FileNotFoundError` in stderr）：eval 线程进入"排水模式"，丢弃后续入队，等 SENTINEL 退出。
- 新增 CLI 参数：`--eval-timeout`（默认 180）、`--no-evaluate`（只生成）。

#### 策略 3：K 路并行生成 + 即时评测（`generate_and_evaluate_parallel.py`）

把策略 2 的串行 gen 换成 `ThreadPoolExecutor(max_workers=K)`，K 默认 2，可用 `$GEN_WORKERS` 或 `--gen-workers` 覆盖。评测消费者仍只有一个线程。

```
ThreadPoolExecutor(max_workers=K)
 ├─ gen op_1 ──► op_1.txt ──► queue.put(op_1)
 ├─ gen op_2 ──► op_2.txt ──► queue.put(op_2)         eval thread (single):
 ├─ ...                                                │  while not SENTINEL:
 └─ gen op_K ──► op_K.txt ──► queue.put(op_K)         │      subprocess.run(eval_single_runner)
                                                       │      persist result.json
                                                       ▼
```

限流处理：K 路池本身是上限；`generate_with_claude_code` 检测到限流标志会静默丢弃输出，主循环看到没文件就不入队，其它 worker 继续。

**实测（K=2，3 个激活算子）**：

| 时间 | 事件 |
|---|---|
| 0s | relu + sigmoid 生成同时启动 |
| 31s | relu 生成完成 → 评测开始 + elu 立刻填补 gen 槽位 |
| 32s | sigmoid 生成完成 → 入队等待 |
| 85s | relu 评测完成 → sigmoid 评测开始 |
| 114s | elu 生成完成 → 入队 |
| 139s | sigmoid 评测完成 → elu 评测开始 |
| 194s | elu 评测完成，全部结束 |

对比：完全串行 308s，1+1 重叠 222s，K=2 并行 194s。

#### 策略 4：多进程并行评测（`evaluation_parallel.py`）

只评测、不生成。每个 worker **独立资源**：

```
ensure_worker_envs(workers, num_npus):
  for i in [0, workers):
    1. 创建 .worker_envs/w{i}/                           # python -m venv --system-site-packages
    2. 克隆 ascend_op_projects/CppExtension/ → ascend_op_projects_w{i}/
       （ignore: dist build *.egg-info，避免跨 Python 版本的 wheel 冲突）
    3. spec[i] = WorkerSpec(worker_id=i, device_id=i % num_npus,
                            op_engineer_dir='ascend_op_projects_w{i}',
                            python_bin='.worker_envs/w{i}/bin/python',
                            venv_bin_dir='.worker_envs/w{i}/bin')

eval_all_parallel:
  ctx = multiprocessing.get_context('spawn')
  manager = ctx.Manager()
  id_queue = manager.Queue() ← 装满 WorkerSpec
  with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                           initializer=_init_worker, initargs=(id_queue,)) as pool:
      futures = {pool.submit(eval_one_op, op, out_dir): op for op in pending}
      for fut in as_completed(futures):
          result[op] = fut.result()
          _persist(output_file, result)     # 原子写
          if fatal: 取消所有 not_running futures，break

eval_one_op (worker 进程):
  ├─ spec = _WORKER_SPEC                                 # initializer 给每个进程灌一个 spec
  ├─ 读 out_dir/{op}.txt
  └─ subprocess.run([spec.python_bin, 'eval_single_runner.py', ...],
                     env={ASCEND_DEVICE_ID, OP_ENGINEER_DIR, PATH=venv_bin:...},
                     timeout=180)
```

NPU 数量自动探测：`/dev/davinci0..N` 个数；Worker `i` 绑到 NPU `i % num_npus`。Worker 数可以大于 NPU 数，但会争抢同一 NPU 上的资源。

```bash
python evaluation_parallel.py \
    --model claude-code --strategy add_shot --categories activation \
    --workers 4 --npus 4
```

### 3.8 串行 vs 并行评测的选择

| 场景 | 推荐 |
|---|---|
| 1 NPU、调试单个算子 | `evaluation.py`（最简单，单一日志流） |
| 1 NPU、生成完成后批量评测 100 个算子 | `evaluation.py`（NPU 单实例本来就只能跑一个，并行没收益） |
| 多 NPU、批量评测 | `evaluation_parallel.py --workers N --npus N` |
| 边生成边评测，gen 串行就够快 | `generate_and_evaluate.py` |
| 边生成边评测，要 K 路并行 gen | `generate_and_evaluate_parallel.py --gen-workers K` |
| 已经生成完了、补评测 | 任何 `evaluation*.py` 都能续跑（读 `result.json` checkpoint） |

---

## 4. 基线性能测量（`generate_baseline_statistics.py`）

测**参考实现 `Model`** 的硬件性能基线，写到 `baselines/{language}_{device}.json`。每个算子在独立 `multiprocessing.Process` 里跑，120s 超时强行回收。

```
for op in dataset:
    p = mp.Process(target=run_op, args=(op, return_dict))
    p.start()
    p.join(timeout=120)
    if p.is_alive(): p.terminate(); result[op] = "timeout"
    else:            result[op] = return_dict[op] or "not supported"
```

`run_op` 内部：

```
1. import backends.ascendc_backend → 注册到 BACKEND_REGISTRY
2. 读 reference/{category}/{op}.py
3. exec ref_src 到 backend.context                       # 注入 Model / get_inputs / ...
4. elapsed_times = backend.time_execution('Model')       # 注意 target 是 Model 不是 ModelNew
5. return_dict[op] = {mean, std, min, max, num_trials, device}
```

异常 → `return_dict[op] = "not supported"`。

```bash
python generate_baseline_statistics.py
# → baselines/ascendc_ai_core-Ascend910B2.json
```

每个算子一个条目，可作为后续 ModelNew 性能加速比的分母。

---

## 5. 整体调用图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  生成路径 A（SDK）                  生成路径 B（Claude CLI 三个变体）      │
│  ─────────────────                  ─────────────────────────────────      │
│  generate_and_write.py              generate_with_claude_code.py           │
│      ↓                              generate_with_cc_and_wiki.py            │
│  prompt_generators.{lang}_{strat}      ↑↑↑（被生成+评测脚本 import 用）   │
│      ↓                                                                     │
│  utils.utils.get_client OR                                                 │
│  utils.utils.get_google_client                                             │
│      ↓                                                                     │
│  client.chat.completions.create                                            │
│      ↓                                                                     │
│  {op}.txt (+ optional {op}_cot.txt / {op}.raw.txt)                         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│              ┌─── evaluation.py (串行)                                     │
│              │                                                             │
│              ├─── generate_and_evaluate.py (1 gen + 1 eval 重叠)           │
│              │       └─ import generate_with_claude_code                    │
│              │                                                             │
│  {op}.txt ──┤                                                              │
│              ├─── generate_and_evaluate_parallel.py (K gen + 1 eval)       │
│              │       └─ import generate_with_claude_code +                  │
│              │          复用 generate_and_evaluate 的 _eval_worker         │
│              │                                                             │
│              └─── evaluation_parallel.py (W workers，独立 venv/NPU)        │
│                                                                            │
│  全部评测路径最终都走：                                                    │
│                                                                            │
│      subprocess: python3 eval_single_runner.py <kernel> <op> <out>         │
│                          ↓                                                 │
│                  utils.evaluation_utils.eval_single                        │
│                          ↓                                                 │
│         backends.ascendc_backend.AscendBackend                             │
│          ├─ compile()              → utils.ascend_compile_pipeline         │
│          │                            (msopgen → build.sh → deploy → pybind)│
│          ├─ correctness_execution()→ utils.correctness.execute_template    │
│          │                            (5 trial × allclose)                 │
│          └─ time_execution()       → utils.performance.time_execution_*    │
│                                       (3 warmup + 100 trial × event)       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

基线（旁路）：
generate_baseline_statistics.py → mp.Process per op → backend.time_execution('Model')
                                                    → baselines/{lang}_{device}.json
```

---

## 6. 典型工作流

### 6.1 SDK 模型测一个类别

```bash
python generate_and_write.py --model deepseek-chat --strategy add_shot --categories activation
python evaluation.py        --model deepseek-chat --strategy add_shot --categories activation
```

### 6.2 Claude CLI 测一个 batch（最快路径）

```bash
bash batch_scripts/batch_01_baseline_concurrent.sh
# 内部其实是：
#   python generate_and_evaluate.py --model-name claude-code --strategy add_shot \
#     --runs 1 --timeout 1200 --eval-timeout 180 --disable-skills --ops <10 个算子>
```

要并行生成：

```bash
GEN_WORKERS=2 python generate_and_evaluate_parallel.py \
    --model-name claude-code --strategy add_shot --disable-skills \
    --ops <从 batch_01_baseline_concurrent.sh 拷过来的 10 个算子>
```

### 6.3 已有 kernel 文件，只跑评测

```bash
# 单 NPU
python evaluation.py --model claude-code --strategy add_shot --categories activation

# 多 NPU
python evaluation_parallel.py --model claude-code --strategy add_shot \
    --categories activation --workers 4 --npus 4
```

### 6.4 三种 prompt 策略对比

跑 `batch_scripts/batch_NN_{baseline,bp,wiki}_concurrent.sh` 三个脚本，分别落盘到 `claude-code/`、`claude-code-bp/`、`claude-code-wiki/` 三个独立目录，事后直接对比 `result.json` 里的 compile/correctness 通过率。

### 6.5 跑基线（一次性，硬件相关）

```bash
python generate_baseline_statistics.py
# → baselines/ascendc_ai_core-Ascend910B2.json
```

### 6.6 断点续跑

所有调度策略都把 `result.json` 当 checkpoint：

- 生成阶段：`{op}.txt` 存在 → skip 生成（`generate_with_claude_code.py:160`）。
- 评测阶段：`result[op]` 已存在 → skip 评测（`evaluation.py:26`、`_eval_worker` 内的 `if op in result: continue`）。
- `result.json` 用 tempfile + `os.replace` 原子写入，Ctrl-C 不会损坏。

直接重跑同一命令即可续上。

---

## 7. 参数与环境变量速查

| 参数 | 默认 | 出现于 | 含义 |
|---|---|---|---|
| `--model` / `--model-name` | `deepseek-chat` / `claude-code` | 全部 | 输出子目录名 |
| `--strategy` | `add_shot` | 全部 | prompt 策略，加载 `prompt_generators.{lang}_{strategy}` |
| `--categories` | `['activation']` 或 `['fuse']` | 全部 | 过滤算子，`all` 表示全量 |
| `--ops` | — | Claude CLI 系列 | 显式算子列表，覆盖 `--categories` |
| `--runs` | 1 | 全部 | 重复运行次数（输出到 `run0`、`run1`...） |
| `--timeout` | 1200 | Claude CLI 系列 | 单算子生成超时（秒） |
| `--eval-timeout` | 180 | generate_and_evaluate(_parallel) | 单算子评测超时（秒） |
| `--disable-skills` | false | Claude CLI 系列 | 透传 `--disable-slash-commands` 给 claude CLI |
| `--with-best-practices` | false | Claude CLI 系列 | prompt 前面接 `ascendc-api-best-practices.md` |
| `--with-wiki` | false | Claude CLI 系列 | 启用 wiki 检索（Read/Glob/Grep 或 cann-ask） |
| `--no-evaluate` | false | generate_and_evaluate(_parallel) | 只生成不评测 |
| `--gen-workers` | 2 | generate_and_evaluate_parallel | K 路并行生成；`$GEN_WORKERS` 可覆盖 |
| `--workers` | 2 | evaluation_parallel | 并行评测 worker 数 |
| `--npus` | auto | evaluation_parallel | NPU 数；默认扫 `/dev/davinci*` |

环境变量：
- `ASCEND_DEVICE_ID`：评测进程绑 NPU id。
- `OP_ENGINEER_DIR`：编译工作区路径（并行评测每 worker 用 `ascend_op_projects_w{i}`）。
- `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`：`ascend_compile` 内部自动维护，不要手动覆盖。
- `GEN_WORKERS`：parallel 生成默认 K。
- `STRATEGY` / `RUNS` / `TIMEOUT` / `EVAL_TIMEOUT`：batch 脚本顶部一律支持。

---

## 8. 故障定位速查

| 现象 | 含义 | 下一步 |
|---|---|---|
| `{op}.raw.txt` 出现但 `{op}.txt` 缺失 | Claude 输出缺少六个变量名之一 | 看 `raw.txt`，多半是模型加了前言或丢了 `model_src` |
| `compile_info` 含 `Exit Code: ...` + `[STDERR]gcc: error: ...` | C++ 编译失败 | 看错误行号；常见 `float ↔ unsigned` 互转、tiling 结构体类型不一致 |
| `compile_info` 含 `msopgen` 报错 | 算子工程模板生成失败 | 检查 `project_json_src` JSON 是否合法、`ascendc_device` 设置 |
| `correctness_info: 'Segmentation fault'` | 评测子进程 SIGSEGV | 看上一个算子是否污染了 opp 部署目录；并行评测确认每 worker 在自己 workspace |
| `correctness_info: '[FAIL] Output shape mismatch'` | ModelNew 输出形状错 | kernel 的输出 shape / strides 与 PyTorch reference 不一致 |
| `correctness_info: '[FAIL] Output mismatch'` | 数值差 > atol=rtol=1e-4 | kernel 数学有误，或 reduce 顺序导致浮点漂移 |
| `correctness_info: 'Timeout fault'` | 评测子进程 180s 内未返回 | 死循环或同步缺失；看 kernel 的 SyncAll / barrier 逻辑 |
| 上层报 `FileNotFoundError` 致命错误 | `config.project_root_path` 配错 | 进入 repo 根再执行；`project_root_path = os.getcwd()` |
| `result.json` 长时间不更新 | 评测子进程还在跑 / SENTINEL 已发但 evaluator 死循环 | 看 `[EVAL]` 日志时间戳；监控 `ps aux | grep eval_single` |
| 并行评测时第一个算子很慢、后续快 | 每 worker 首次跑要 `pip install custom_ops_lib` | 正常现象；只发生在 `.worker_envs/w{i}` 还没装算子库时 |
| 生成阶段 `[SKIP]` 频繁出现 | Claude CLI 命中限流 | 降低 `--gen-workers`、加 `--timeout`、或者改用 SDK 路径 |
