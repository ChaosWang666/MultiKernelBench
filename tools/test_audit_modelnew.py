"""audit_modelnew 正反例测试 — 针对 MultiKernelBench 当前 ModelNew 形态"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.ast_audit import audit_modelnew


CASES = [
    # ===== 正例 =====
    {
        "name": "POS - clean ModelNew (Parameter only + single-line forward)",
        "src": """
import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm_weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
    def forward(self, x):
        return custom_ops_lib.gemm_add_relu_custom(
            x.contiguous(), self.gemm_weight.contiguous(), self.bias.contiguous(),
        )
""",
        "op_name": "gemm_add_relu",
        "expect_pass": True,
    },
    {
        "name": "POS - with register_buffer (causal_mask)",
        "src": """
import torch
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len):
        super().__init__()
        self.in_proj_weight = torch.nn.Parameter(torch.randn(3 * d_model, d_model))
        self.register_buffer('causal_mask', torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())
    def forward(self, x):
        return custom_ops_lib.causal_attention_custom(x, self.in_proj_weight, self.causal_mask)
""",
        "op_name": "causal_attention",
        "expect_pass": True,
    },
    {
        "name": "POS - nn.Parameter without torch. prefix",
        "src": """
import torch
from torch import nn
import custom_ops_lib

class ModelNew(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d))
    def forward(self, x):
        return custom_ops_lib.foo_custom(x, self.w)
""",
        "op_name": "foo",
        "expect_pass": True,
    },

    # ===== 反例: __init__ =====
    {
        "name": "NEG - __init__ uses nn.Linear (current MKB hack)",
        "src": """
import torch
import torch.nn as nn
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = torch.nn.Linear(in_features, out_features, bias=False)
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
    def forward(self, x):
        return custom_ops_lib.gemm_add_relu_custom(x, self.gemm.weight, self.bias)
""",
        "op_name": "gemm_add_relu",
        "expect_pass": False,
        "expect_rule": "init_nn_module",
    },
    {
        "name": "NEG - __init__ uses nn.MultiheadAttention",
        "src": """
import torch
import torch.nn as nn
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
    def forward(self, x):
        return custom_ops_lib.mha_custom(x, self.mha.in_proj_weight)
""",
        "op_name": "mha",
        "expect_pass": False,
        "expect_rule": "init_nn_module",
    },
    {
        "name": "POS - __init__ scalar attr (multiplier, eps)",
        "src": """
import torch
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.multiplier = multiplier             # 标量 init_attr
        self.negative_slope = negative_slope     # 标量 init_attr
        self.eps = 1e-5                          # 字面量
    def forward(self, x):
        return custom_ops_lib.foo_custom(x, self.weight, self.bias,
                                         self.multiplier, self.negative_slope, self.eps)
""",
        "op_name": "foo",
        "expect_pass": True,
    },
    {
        "name": "NEG - __init__ assigns raw Tensor without nn.Parameter wrap",
        "src": """
import torch
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.x = torch.randn(d)             # ❌ 裸 Tensor 不是 Parameter, 也不是 buffer
        self.w = torch.nn.Parameter(torch.randn(d))
    def forward(self, x):
        return custom_ops_lib.foo_custom(x, self.w)
""",
        "op_name": "foo",
        "expect_pass": False,
        "expect_rule": "init_unknown_assign",
    },

    # ===== 反例: forward =====
    {
        "name": "NEG - forward multiple statements",
        "src": """
import torch
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.contiguous()
        return custom_ops_lib.foo_custom(y)
""",
        "op_name": "foo",
        "expect_pass": False,
        "expect_rule": "forward_not_single_return",
    },
    {
        "name": "NEG - forward calls torch.matmul",
        "src": """
import torch
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(d, d))
    def forward(self, x):
        return torch.matmul(x, self.w)
""",
        "op_name": "foo",
        "expect_pass": False,
        "expect_rule": "forward_wrong_target",
    },
    {
        "name": "NEG - forward calls wrong op name",
        "src": """
import torch
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return custom_ops_lib.something_else_custom(x)
""",
        "op_name": "foo",
        "expect_pass": False,
        "expect_rule": "forward_wrong_op",
    },
    {
        "name": "NEG - forward arg uses .matmul() method",
        "src": """
import torch
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(d, d))
    def forward(self, x):
        return custom_ops_lib.foo_custom(x.matmul(self.w))
""",
        "op_name": "foo",
        "expect_pass": False,
        "expect_rule": "forward_arg_invalid",
    },
    {
        "name": "NEG - forward has if branch",
        "src": """
import torch
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if x.shape[0] > 1024:
            return custom_ops_lib.foo_custom(x)
        return x
""",
        "op_name": "foo",
        "expect_pass": False,
        "expect_rule": "forward_not_single_return",
    },
    {
        "name": "NEG - missing class ModelNew",
        "src": """
import torch
def foo(x):
    return x
""",
        "op_name": "foo",
        "expect_pass": False,
        "expect_rule": "missing_class",
    },
]


def run():
    passed = failed = 0
    for case in CASES:
        try:
            vios = audit_modelnew(case["src"], case["op_name"])
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
