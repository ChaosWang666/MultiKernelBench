# MultiKernelBench 防 Hack 加固

针对 LLM agent 在算子生成评测中常见的 hack 行为 (代算 / zero-fill stub / 接口偷工 /
参数忽略 等), 给评测流程加了**4 层纵深防护** + 配套 **prompt 加固**。

## 加固了哪些能力

| 层 | 能力 | 防住的典型 hack | 状态 |
|---|---|---|---|
| **解析** | 校验 agent 输出的 `.txt` 是否完整 (6 个必需字符串变量 project_json_src / host_tiling_src / host_operator_src / kernel_src / python_bind_src / model_src) | 输出格式残缺 | 已实现 |
| **接口校验** | 从 reference `class Model` 自动推 schema (参数清单 / shape / dtype), 校验 `ModelNew` 是否覆盖所有 weight/buffer 参数 | 参数忽略、接口偷工 (reference 里的 `nn.Linear` 权重在 ModelNew 里没声明) | 已实现 |
| **静态检查** | 扫 `model_src` 的 AST: `__init__` 只允许 `nn.Parameter` / `register_buffer` / 标量 attr; `forward` 必须单行 `return custom_ops_lib.<op>_custom(...)`; 同时扫 `kernel_src` / `python_bind_src` 关键字黑名单 (`at::matmul` / `aclnn*` 等明文 C++ 代算) | `__init__` 里塞 `nn.Linear`; `forward` 多行 + 调 `torch.matmul`; C++ 里明文调 ATen / ACLNN | 已实现 |
| **运行时拦截** | 用 PyTorch `TorchFunctionMode` + meta tensor dry-run forward, 任何非内存调整方法的 PyTorch 调用 → 失败 | 动态绕过 (`getattr(torch, 'matmul')`) / 别名 / helper 模块 / 自定义 op 内嵌套 PyTorch | 已实现 |
| **设备 kernel 黑名单 (profiler)** | NPU 上 profiler 记录设备真实 launch 了哪些 kernel, 黑名单 (含 `Conv2D` / `MatMul` / `GroupNorm` / `aclnnConv*` 等"代算 kernel") 命中即判 hack | 绕过 PyTorch dispatcher 直接调 ACLNN / 厂商加速库的设备层代算 | **未实现** (NPU 真机依赖) |
| **Prompt 加固** | 默认 ascendc strategy 已切换为 anti-hack 模板, 含规则段 + 反例 + 自动嵌入的接口 schema | 引导 agent 不要复制 reference 的 PyTorch 实现 | 已实现 |

设计思路: 保留 MultiKernelBench 现有 `class ModelNew` + 输出格式 + `custom_ops_lib`
namespace, 通过**严格约束 ModelNew 结构 + AST + 运行时拦截**达成防 hack 目标。
Backend / 编译流水线 / 精度对比逻辑**完全不动**。

---

## 怎么用

### 1. 批量检查一个提交目录

```bash
cd MultiKernelBench
python tools/check_hack.py <submissions_dir>
```

工具会:
- 扫提交目录下所有 `*.txt` (用户提交了什么就检测什么)
- 用文件名 `<op_name>.txt` 在 `reference/<category>/<op_name>.py` 找对应 reference
- 逐项检查, 输出汇总表 + 失败原因

### 2. 输出样例

```
OP                                     RESULT  ISSUES
gemm_add_relu                          FAIL    2
gemm_sigmoid_scaling_residual_add      FAIL    1
multi_query_attention                  PASS    -

===== SUMMARY =====
  PASS:  1 / 3
  FAIL:  2 / 3
  SKIP:  0 / 3

===== Failures =====

[gemm_add_relu]
  - line 9: __init__ 实例化了 nn.Module 子类 (torch.nn.Linear), 只允许 nn.Parameter / register_buffer
  - line 14: forward 第 2 个参数访问了嵌套属性 self.gemm.weight, 不允许

[gemm_sigmoid_scaling_residual_add]
  - ModelNew 实例化失败: nn.Parameter 收到的 shape 参数类型不对 (期望 tuple, 实际 float)
```

### 3. 命令行参数

| 参数 | 说明 |
|---|---|
| `<submissions_dir>` | 提交目录 (必填) |
| `--ref-root <path>` | reference 树根目录 (默认 `<MKB>/reference`) |
| `--op <name>` | 只检测指定算子 |
| `-v` | 详细输出 (含 traceback) |

退出码: `0` 全过 / `1` 有 FAIL/ERROR。

### 4. Prompt 默认就是加固版

```python
from prompt_generators import ascendc_anti_hack
from prompt_generators.prompt_registry import PROMPT_REGISTRY

prompt = PROMPT_REGISTRY['ascendc']['anti_hack'].generate('multi_query_attention')
```

`evaluation.py` / `generate_and_write.py` 的 `--strategy` 默认值兼容旧名 `add_shot`,
内部已经切到 anti-hack 模板, 无需改调用方式。

---

## 限制

- L3 默认 dry-run 模式不需要 NPU, 但 wrapper 内的 C++ 代码 (kernel/python_bind) 实际没执行,
  C++ 层的 ATen / ACLNN 调用要靠静态扫描 (粗) + 真机评测时的 L4 profiler 黑名单 (终极兜底)
- L4 依赖 NPU 真机和 profiler 数据, 当前未实现
- Backend / `correctness.py` / 编译流水线本次未改, 精度对比仍走原有 `torch.allclose` 流程
