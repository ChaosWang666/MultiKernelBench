"""AST audit 正反例测试。

正例: 干净的 wrapper, 应通过
反例: 各种 hack 形态, 应被检测出
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.ast_audit import audit_submission, format_violations


CASES = [
    # ===== 正例 =====
    {
        "name": "POS-clean wrapper (single op call)",
        "src": """
import torch
def gemm_add_relu(x, bias, gemm_weight, in_features, out_features, bias_shape):
    return torch.ops.mkb.gemm_add_relu(x, bias, gemm_weight, in_features, out_features, bias_shape)
""",
        "fn": "gemm_add_relu",
        "expect_pass": True,
    },
    {
        "name": "POS-allowed memory ops (.contiguous .to .view)",
        "src": """
import torch
def my_op(x, w):
    x = x.contiguous().to(torch.float16)
    w = w.view(-1, 64)
    return torch.ops.mkb.my_op(x, w)
""",
        "fn": "my_op",
        "expect_pass": True,
    },
    {
        "name": "POS-pure python integer arithmetic in kwargs",
        "src": """
import torch
def my_op(x, stride):
    s = stride * 2 + 1   # 纯 Python 整数算术
    return torch.ops.mkb.my_op(x, s)
""",
        "fn": "my_op",
        "expect_pass": True,
    },

    # ===== 反例: torch.* 计算函数 =====
    {
        "name": "NEG-torch.matmul",
        "src": """
import torch
def my_op(x, w, bias):
    y = torch.matmul(x, w)
    return torch.ops.mkb.my_op(y, bias)
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "torch_compute_call",
    },
    {
        "name": "NEG-torch.softmax + torch.gelu",
        "src": """
import torch
def my_op(x):
    y = torch.gelu(x)
    return torch.softmax(y, dim=-1)
""",
        "fn": "my_op",
        "expect_pass": False,
    },

    # ===== 反例: torch.nn.functional =====
    {
        "name": "NEG-F.conv2d",
        "src": """
import torch
import torch.nn.functional as F
def my_op(x, w, bias):
    return F.conv2d(x, w, bias)
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "torch_nn_functional",
    },
    {
        "name": "NEG-torch.nn.functional.gelu (full path)",
        "src": """
import torch
def my_op(x):
    return torch.nn.functional.gelu(x)
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "torch_nn_functional",
    },

    # ===== 反例: nn 模块实例化 =====
    {
        "name": "NEG-nn.Linear instantiated in body",
        "src": """
import torch
import torch.nn as nn
def my_op(x, w):
    layer = nn.Linear(64, 64)
    return layer(x)
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "torch_nn_module",
    },

    # ===== 反例: Tensor 计算方法 =====
    {
        "name": "NEG-tensor method .matmul / .softmax / .sum",
        "src": """
import torch
def my_op(x, w):
    y = x.matmul(w)
    y = y.softmax(dim=-1)
    return y.sum()
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "tensor_compute_method",
    },

    # ===== 反例: @ 运算符 =====
    {
        "name": "NEG-@ matmul operator",
        "src": """
import torch
def my_op(x, w):
    return x @ w
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "tensor_matmul_operator",
    },

    # ===== 反例: if 分支 (fallback hack) =====
    {
        "name": "NEG-if fallback",
        "src": """
import torch
def my_op(x, w):
    if x.shape[0] == 8192:
        return torch.ops.mkb.my_op(x, w)
    else:
        return torch.matmul(x, w)
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "if_branch",
    },

    # ===== 反例: 函数体内 import =====
    {
        "name": "NEG-inline import torch.nn.functional",
        "src": """
import torch
def my_op(x):
    import torch.nn.functional as F
    return F.gelu(x)
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "inline_import",
    },

    # ===== 反例: exec/getattr 动态绕过 (但 L2 只能拦明文) =====
    {
        "name": "NEG-exec",
        "src": """
import torch
def my_op(x):
    exec("x = x + 1")
    return x
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "forbidden_builtin",
    },
    {
        "name": "NEG-getattr (动态名字解析)",
        "src": """
import torch
def my_op(x):
    fn = getattr(torch, "matmul")
    return fn(x, x)
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "forbidden_builtin",
    },

    # ===== 反例: 错误的 op_ns =====
    {
        "name": "NEG-wrong torch.ops namespace",
        "src": """
import torch
def my_op(x, w):
    return torch.ops.someother_ns.my_op(x, w)
""",
        "fn": "my_op",
        "expect_pass": False,
        "expect_rule": "forbidden_op_ns",
    },
]


def run():
    passed, failed = 0, 0
    for case in CASES:
        try:
            vios = audit_submission(case["src"], case["fn"], allowed_op_ns="mkb")
        except Exception as e:
            print(f"  ERR  {case['name']}: {type(e).__name__}: {e}")
            failed += 1
            continue

        is_pass = (len(vios) == 0)
        expect = case["expect_pass"]

        if is_pass == expect:
            tag = " OK "
            passed += 1
            extra = ""
            if not expect:
                # 还得检查命中规则正确
                expect_rule = case.get("expect_rule")
                if expect_rule and not any(v.rule == expect_rule for v in vios):
                    tag = "WARN"
                    extra = f"  [expected rule {expect_rule!r}, got {[v.rule for v in vios]}]"
            print(f"  {tag}  {case['name']}{extra}")
        else:
            tag = "FAIL"
            failed += 1
            print(f"  {tag}  {case['name']}")
            print(f"        expected_pass={expect}, got_violations={len(vios)}")
            for v in vios[:3]:
                print(f"          - {v}")

    print(f"\n=== {passed}/{len(CASES)} OK, {failed} FAIL ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
