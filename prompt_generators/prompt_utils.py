import os
from config import project_root_path
from dataset import dataset
from utils.utils import read_file, underscore_to_pascalcase

template_statement="""You write custom {} kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom {} kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
template_instruction="""
Optimize the architecture named Model with custom {} operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

template_example_intro='''
Here's an example to show you the syntax of inline embedding custom {} operators in torch: The example given architecture is:
'''

template_new_arch_intro='''
The example new arch with custom {} kernels looks like this:
'''

ASCENDC_PROBLEM_STATEMENT = 'You are an expert in writing custom AscendC kernels to optimize PyTorch architectures by replacing specific operators for performance gains.\n'
ASCENDC_PROBLEM_INSTRUCTION='''
Your task: Replace relevant PyTorch operators in the architecture named Model with custom AscendC kernels. Generate an optimized version named ModelNew, including the six Python strings listed above. Just output the code, no other text, and NO testing code!\n
'''

def read_relavant_files(language, op, example):
    category = dataset[op]['category']
    example_arch_path = os.path.join(
        project_root_path, f"prompts/cuda_model_{example}.py"
    )
    example_new_arch_path = os.path.join(
        project_root_path, f"prompts/{language}_new_model_{example}.py"
    )
    new_arch_path = os.path.join(
        project_root_path, f"reference/{category}/{op}.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )
    if not os.path.exists(new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {new_arch_path}"
        )
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    arch = read_file(new_arch_path)
    return arch, example_arch, example_new_arch

def generate_template(arc_src, example_arch_src, example_new_arch_src, language):
    prompt = template_statement.format(language, language)

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        {template_example_intro.format(language)} \n
        ``` \n
        {example_arch_src}
        ``` \n
        {template_new_arch_intro.format(language)} 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += template_instruction.format(language)
    return prompt    

def ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, example_op):
        # add custom to name to prevent conficts with existing operators
        op = op + '_custom'
        example_op = example_op + '_custom'
        prompt = ASCENDC_PROBLEM_STATEMENT

        if example_arch_src != "" and example_new_arch_src != "":
            prompt += f"""
    Here is an example to illustrate the expected transformation using custom AscendC operators. **Original architecture with kernel name `{example_op}`:**\n
    ```python \n
    {example_arch_src}
    ``` \n
    Transformed version using custom AscendC kernels:
    This transformation includes six embedded Python strings: `project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src` and `model_src`.
    The kernel function name in `kernel_src` must exactly match the provided kernel name. The operator definition in `project_json_src` and `host_operator_src` should also correspond to the kernel name, but follow PascalCase naming:
    ```python
    {example_new_arch_src}
    ``` \n
    """

        prompt += f"""
    Now, you are given the following architecture with kernel name {op}(PascalCase: {underscore_to_pascalcase(op)}): \n
    ```python
    {arc_src}
    ```
        """
        prompt += ASCENDC_PROBLEM_INSTRUCTION
        return prompt


# ===========================================================================
# Anti-hack prompt template
# ===========================================================================
# 在原 ascendc_template 基础上新增:
#  - ANTI-HACK RULES 段 (明示 L2/L3 检查规则)
#  - ALGORITHM SPEC 段 (含 reference + 自动嵌入的 schema 签名)
#  - ONE-SHOT 段 (用一个干净融合算子样板演示 ModelNew 形态)
#  - COUNTER-EXAMPLES 段 (3 个典型 hack 反例)
#
# 与原 template 兼容: 输出仍是 6 个字符串变量, 编译流水线不变。

ASCENDC_ANTI_HACK_RULES = """
[ANTI-HACK RULES — strictly enforced; ANY violation results in 0 score]

The evaluator runs static AST checks (L2) and runtime dispatch interception (L3)
on your `ModelNew`. The structure MUST satisfy:

A. `__init__` body is allowed to contain only:
   1. `super().__init__()`
   2. `self.<name> = (torch.)?nn.Parameter(torch.randn(...))`
      — Every learnable weight (including the ones inside the reference's
        `nn.Linear` / `nn.Conv*` / `nn.LayerNorm` / `nn.MultiheadAttention` etc.)
        MUST be declared this way as `nn.Parameter`.
   3. `self.register_buffer(name, ...)`
   FORBIDDEN:
   - Instantiating ANY `nn.<Module>` subclass (e.g. `nn.Linear`, `nn.Conv2d`,
     `nn.LayerNorm`, `nn.MultiheadAttention`, `nn.GroupNorm`). Only `nn.Parameter`
     is allowed.
   - Any `torch.<compute>` / `F.*` calls.

B. `forward` MUST be a single `return` statement:
       def forward(self, <inputs>):
           return custom_ops_lib.<op_name>_custom(<arg1>, <arg2>, ...)
   Each <argN> may only be:
   - a forward input parameter (e.g. `x`)
   - `self.<name>` (declared in `__init__`)
   - either of the above followed by `.contiguous()` / `.to(...)` / `.view(...)` /
     `.reshape(...)` / `.unsqueeze(...)` / `.squeeze(...)`
   FORBIDDEN inside `forward`:
   - Any intermediate variable assignment (`y = ...`)
   - Any `if` / `for` / `while`
   - Any `torch.*` / `nn.*` / `F.*` call
   - Any tensor arithmetic (`x + y`, `x * w`, `x @ w`)
   - Any tensor compute method (`x.matmul(y)`, `x.softmax()`, `x.sum()`, ...)

C. The AscendC kernel in `kernel_src` MUST implement the full operator semantics
   end-to-end. The following are also detected as hacks and scored 0:
   - Zero-fill stub (e.g. `AscendC::Duplicate(out, 0, length)` only)
   - Implementing only a tail subset of the operator (e.g. only the final clamp/relu)
   - Ignoring any declared weight/bias parameter inside the kernel
"""

ASCENDC_ANTI_HACK_INSTRUCTION = """
[OUTPUT REQUIREMENTS]
Output exactly 6 Python string variables with these EXACT names:
    project_json_src, host_tiling_src, host_operator_src,
    kernel_src, python_bind_src, model_src

Output ONLY the code (in a single python code block). No prose, no testing code.
The reference `class Model` is shown only so you understand the operator semantics —
copying any PyTorch operation from `Model.forward` into your `ModelNew.forward`
will be detected by the AST/runtime checks and scored 0.
"""


def _build_schema_block(arc_src, op):
    """Try to auto-extract schema from the reference module text and render it."""
    try:
        # Lazy import 防止循环依赖 / 防止 utils 路径未挂上 sys.path 时崩溃
        import sys
        import os
        import tempfile
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from utils.schema_extractor import extract_schema

        # extract_schema 接收文件路径,这里把 arc_src 写到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False, encoding='utf-8') as tf:
            tf.write(arc_src)
            tmp_path = tf.name
        try:
            schema = extract_schema(tmp_path, op, "")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        lines = [
            "[ALGORITHM SCHEMA — auto-extracted from reference Model]",
            f"Operator: {op}",
            f"Signature: {schema.call_signature}",
            "",
            "Required parameters in your ModelNew (declare each as `nn.Parameter`):",
        ]
        if schema.parameters:
            for p in schema.parameters:
                lines.append(f"  - self.{p.name}: {p.dtype}{list(p.shape)}  "
                             f"(reference internal name: {p.raw_name})")
        else:
            lines.append("  (no learnable parameters)")

        if schema.buffers:
            lines.append("")
            lines.append("Required buffers (declare via `register_buffer`):")
            for p in schema.buffers:
                lines.append(f"  - {p.name}: {p.dtype}{list(p.shape)}  "
                             f"(reference internal name: {p.raw_name})")

        lines.append("")
        lines.append("Forward inputs:")
        for p in schema.inputs:
            if p.shape is not None:
                lines.append(f"  - {p.name}: {p.dtype}{list(p.shape)}")
            else:
                lines.append(f"  - {p.name}: {p.py_type or '?'}")

        if schema.outputs:
            lines.append("")
            lines.append("Output:")
            for p in schema.outputs:
                lines.append(f"  - {p.name}: {p.dtype}{list(p.shape)}")
        return "\n".join(lines)
    except Exception as e:
        return f"[ALGORITHM SCHEMA — auto-extraction failed: {type(e).__name__}: {e}]"


def ascendc_anti_hack_template(arc_src, example_arch_src, example_new_arch_src,
                                op, example_op):
    """Anti-hack 加固版 ascendc prompt 模板。

    与原 ascendc_template 兼容: 输出仍是 6 个字符串变量, 编译流水线不变。
    新增 4 段: anti-hack rules, schema, one-shot, counter-examples。
    """
    op_custom = op + '_custom'
    example_op_custom = example_op + '_custom'

    prompt = ASCENDC_PROBLEM_STATEMENT
    prompt += ASCENDC_ANTI_HACK_RULES

    # ---- ONE-SHOT ----
    # 只展示 correct 形态;不再贴 example 的 reference Model
    # (reference->ModelNew 的转换关系由反例段 + 目标算子的 SPEC 段共同给出)
    if example_new_arch_src:
        prompt += f"""
[ONE-SHOT EXAMPLE — clean submission for `{example_op_custom}`]

Use this as a structural template: note how every learnable weight is declared
as `nn.Parameter` in `__init__` (NOT `nn.Linear` / `nn.Conv*` / etc.), and
`forward` is a single `return custom_ops_lib.<op>_custom(...)` line.

```python
{example_new_arch_src}
```
"""

    # ---- COUNTER-EXAMPLES ----
    prompt += """
[COUNTER-EXAMPLES — these will be detected and scored 0]

# COUNTER 1: __init__ uses nn.Linear → forbidden nn.Module instantiation
class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gemm = torch.nn.Linear(in_features, out_features)   # FORBIDDEN
    def forward(self, x):
        return custom_ops_lib.foo_custom(x, self.gemm.weight)

# COUNTER 2: forward does PyTorch compute (代算)
class ModelNew(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(...))
    def forward(self, x):
        y = torch.matmul(x, self.w.T)                           # FORBIDDEN
        return torch.nn.functional.relu(y + self.bias)          # FORBIDDEN

# COUNTER 3: kernel is a zero-fill stub  (in kernel_src)
extern "C" __global__ __aicore__ void foo_custom(...) {
    AscendC::Duplicate(out, 0.0f, length);                       // FORBIDDEN
}
"""

    # ---- ALGORITHM SPEC + SCHEMA ----
    prompt += f"""
[ALGORITHM SPEC]

Reference `class Model` (use ONLY to understand the operator semantics; do NOT
copy any PyTorch op from its forward into your ModelNew.forward):
```python
{arc_src}
```

"""
    prompt += _build_schema_block(arc_src, op)

    prompt += f"""

Your task: implement operator `{op_custom}` (PascalCase: {underscore_to_pascalcase(op_custom)}).
"""
    prompt += ASCENDC_ANTI_HACK_INSTRUCTION
    return prompt