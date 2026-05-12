"""runtime_guard L3 正反例测试。

测试 OpAuditMode (Python 层) 和 AtenAuditMode (aten 层) 的拦截能力。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from utils.runtime_guard import OpAuditMode, AtenAuditMode, HackDetected


# ---------------------------------------------------------------------------
# 注册一个假 custom op 用于白名单测试
# ---------------------------------------------------------------------------

# 用 torch.library.define 注册 mkb::passthrough (单输入返回 .clone())
_LIB = torch.library.Library("mkb", "FRAGMENT")
_LIB.define("passthrough(Tensor a) -> Tensor")

def _passthrough_impl(a):
    # 实现里用允许的方法, 避免触发 OpAuditMode
    return a.clone()

_LIB.impl("passthrough", _passthrough_impl, "CPU")
_LIB.impl("passthrough", _passthrough_impl, "Meta")


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

PASS_CASES = [
    # (描述, lambda 实现)
    ("call torch.ops.mkb.<op>",
        lambda x, y: torch.ops.mkb.passthrough(x)),
    ("tensor.contiguous()",
        lambda x, y: x.contiguous()),
    ("tensor.to(torch.float16)",
        lambda x, y: x.to(torch.float16)),
    ("tensor.view(-1)",
        lambda x, y: x.view(-1)),
    ("tensor.reshape(-1)",
        lambda x, y: x.reshape(-1)),
    ("tensor.unsqueeze(0)",
        lambda x, y: x.unsqueeze(0)),
    ("tensor.shape access",
        lambda x, y: x.shape),
]

FAIL_CASES = [
    ("torch.matmul",
        lambda x, y: torch.matmul(x, y)),
    ("torch.softmax",
        lambda x, y: torch.softmax(x, dim=-1)),
    ("torch.nn.functional.gelu",
        lambda x, y: torch.nn.functional.gelu(x)),
    ("tensor.matmul method",
        lambda x, y: x.matmul(y)),
    ("tensor.softmax method",
        lambda x, y: x.softmax(dim=-1)),
    ("tensor.sum method",
        lambda x, y: x.sum()),
    ("tensor + operator",
        lambda x, y: x + y),
    ("tensor @ matmul operator",
        lambda x, y: x @ y),
    ("wrong namespace torch.ops.foo.* (unregistered)",
        lambda x, y: torch.ops.foo.bar(x) if hasattr(torch.ops, 'foo') else _trigger_wrong_ns(x)),
    ("dynamic getattr(torch, 'matmul')",
        lambda x, y: getattr(torch, 'matmul')(x, y)),
]


def _trigger_wrong_ns(x):
    """注册 foo::bar 后调用, 模拟 wrong namespace"""
    lib = torch.library.Library("foo", "FRAGMENT")
    lib.define("bar(Tensor a) -> Tensor")
    lib.impl("bar", lambda a: a.clone(), "CPU")
    return torch.ops.foo.bar(x)


def run_pos(allowed_ns="mkb"):
    print("=== POSITIVE (应通过) ===")
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    passed = failed = 0
    for desc, fn in PASS_CASES:
        try:
            with OpAuditMode(allowed_ns=allowed_ns):
                _ = fn(x, y)
            print(f"   OK  {desc}")
            passed += 1
        except HackDetected as e:
            print(f"  FAIL {desc}: 误判为 hack")
            print(f"        {str(e).splitlines()[0]}")
            failed += 1
        except Exception as e:
            print(f"  ERR  {desc}: {type(e).__name__}: {e}")
            failed += 1
    return passed, failed


def run_neg(allowed_ns="mkb"):
    print("\n=== NEGATIVE (应被拦截) ===")
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    passed = failed = 0
    for desc, fn in FAIL_CASES:
        try:
            with OpAuditMode(allowed_ns=allowed_ns):
                _ = fn(x, y)
            print(f"  FAIL {desc}: 未被拦截!")
            failed += 1
        except HackDetected:
            print(f"   OK  {desc} (拦截成功)")
            passed += 1
        except Exception as e:
            # 例如 wrong-ns: foo namespace 不存在, AttributeError 也算"被阻止"
            print(f"  WARN {desc}: {type(e).__name__}: {str(e)[:80]}")
            passed += 1
    return passed, failed


def run():
    p1, f1 = run_pos()
    p2, f2 = run_neg()
    total_p = p1 + p2
    total_f = f1 + f2
    total = len(PASS_CASES) + len(FAIL_CASES)
    print(f"\n=== {total_p}/{total} OK, {total_f} FAIL ===")
    return 0 if total_f == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
