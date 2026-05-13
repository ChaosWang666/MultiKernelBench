"""运行时 dispatch 拦截 — L3 防 hack 检查。

入口: with OpAuditMode(allowed_ns="mkb"): submission_func(...)
       (可选叠加) with AtenAuditMode(allowed_ns="mkb"): ...

核心思路:
- TorchFunctionMode 拦截 Python 层调用 (torch.matmul / x + y / x.softmax 等)
- TorchDispatchMode 兜底拦截 aten 层 (防 torch._C._nn.* 等绕过)
- 白名单只有 torch.ops.<allowed_ns>.* + 必要的内存调整 op
- 命中黑名单 raise HackDetected, 给出具体调用名

L3 与 L2 (AST) 的关系:
- L2 在编译期扫描源码, 抓明文 PyTorch 调用
- L3 在运行期截获实际发生的调用, 抓动态绕过 (getattr / exec / helper 模块 / 自定义 op 内嵌套)
- L4 (profiler 黑名单) 兜底设备层 (绕开 PyTorch dispatcher 的 ACLNN 直 launch)
"""

from __future__ import annotations

from typing import Any, Optional, Set

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode


class HackDetected(RuntimeError):
    """运行时拦截到非法调用"""
    pass


# ---------------------------------------------------------------------------
# Python 层白名单 (TorchFunctionMode)
# ---------------------------------------------------------------------------

# 允许的 Tensor 方法 (内存形态调整, 不参与计算)
# __torch_function__ 收到的 func 可能是 torch.Tensor.contiguous 这种, 也可能是
# torch._C._TensorBase.contiguous 之类。我们用方法名 (qualname 末段) 做匹配。
_PYTHON_ALLOWED_METHOD_NAMES: Set[str] = {
    'contiguous', 'to', 'view', 'reshape', 'clone', 'detach',
    'unsqueeze', 'squeeze', 'expand', 'expand_as',
    'size', 'shape', 'dim', 'ndim', 'dtype_property', 'device_property',
    'is_contiguous', 'is_cuda', 'numel', 'element_size',
    'as_strided', 'flatten', 't', 'mT',
    'data_ptr', 'storage_offset', 'stride', 'storage',
    'requires_grad', 'requires_grad_',
    '__repr__', '__str__', '__format__', '__hash__',
    '__getattr__', '__setattr__', '__getattribute__',
    '__bool__', '__int__', '__float__', '__index__', '__len__',
    '__iter__', '__contains__',
    '__deepcopy__', '__reduce_ex__',
}

# 允许的 torch.* (顶层 namespace) 函数, 不涉及实际计算
_PYTHON_ALLOWED_TORCH_FUNCS: Set[str] = {
    'is_tensor', 'is_floating_point', 'is_complex',
    'get_default_dtype', 'set_default_dtype',
    'numel', 'broadcast_shapes',
}


def _func_qualname(func: Any) -> str:
    """拿到 func 的 qualified name, 用于人类可读的报错"""
    mod = getattr(func, '__module__', '') or ''
    qn = getattr(func, '__qualname__', '') or getattr(func, '__name__', '') or repr(func)
    if mod:
        return f"{mod}.{qn}"
    return qn


def _is_allowed_op_call(func: Any, allowed_ns: str) -> bool:
    """判断 func 是不是 torch.ops.<allowed_ns>.* 调用"""
    # 路径 1: OpOverload / OpOverloadPacket 对象有 namespace 属性
    ns = getattr(func, 'namespace', None)
    if ns == allowed_ns:
        return True
    # 路径 2: __torch_function__ 收到的可能是 PyCapsule (C 扩展函数),
    # 它的 __module__ 形如 "torch._ops.mkb"
    mod = getattr(func, '__module__', '') or ''
    if mod == f"torch._ops.{allowed_ns}" or mod.startswith(f"torch._ops.{allowed_ns}."):
        return True
    # 路径 3: 兜底用 qualname 字符串匹配
    qn = _func_qualname(func)
    return f".ops.{allowed_ns}." in qn or qn.startswith(f"torch.ops.{allowed_ns}.")


def _is_allowed_tensor_method(func: Any) -> bool:
    """判断 func 是否 Tensor 上允许的方法 (.contiguous / .to / .view / ...)"""
    name = getattr(func, '__name__', '') or ''
    if name in _PYTHON_ALLOWED_METHOD_NAMES:
        return True
    # property 描述符访问: x.shape / x.dtype / x.device 触发 getset_descriptor.__get__
    # 类似的描述符方法都属于元信息访问, 不参与计算
    if name in ('fget', 'fset', '__get__', '__set__'):
        return True
    # 描述符类型本身 (如 getset_descriptor, member_descriptor)
    cls_name = type(func).__name__
    if cls_name in ('getset_descriptor', 'member_descriptor', 'method_descriptor'):
        # method_descriptor 不一定都是元信息, 但调用时会带具体 method 名,
        # 这里只对 getset/member 直接放行
        if cls_name in ('getset_descriptor', 'member_descriptor'):
            return True
    return False


def _is_allowed_torch_func(func: Any) -> bool:
    """判断 func 是否 torch.* 顶层允许的非计算函数"""
    name = getattr(func, '__name__', '') or ''
    return name in _PYTHON_ALLOWED_TORCH_FUNCS


# ---------------------------------------------------------------------------
# Python 层拦截
# ---------------------------------------------------------------------------

class OpAuditMode(TorchFunctionMode):
    """拦截 Python 层所有 torch op 调用, 白名单之外抛 HackDetected。

    覆盖:
    - torch.<x>(...)              → 检 _PYTHON_ALLOWED_TORCH_FUNCS / 计算函数禁
    - x.<method>(...)             → 检 _PYTHON_ALLOWED_METHOD_NAMES
    - x + y / x * y / x @ y       → __add__/__mul__/__matmul__ 都被拦
    - torch.ops.<ns>.<op>(...)    → 仅 ns == allowed_ns 时放行
    - torch.nn.functional.<x>     → 内部走 aten, 也会被拦

    用法:
        with OpAuditMode(allowed_ns="mkb"):
            output = submission_func(**kwargs)
    """

    def __init__(self, allowed_ns: str = "mkb",
                 extra_allowed_funcs: Optional[Set[str]] = None):
        super().__init__()
        self.allowed_ns = allowed_ns
        self.extra_allowed = set(extra_allowed_funcs or ())

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if self._is_allowed(func):
            return func(*args, **kwargs)

        qn = _func_qualname(func)
        raise HackDetected(
            f"[L3-TorchFunction] submission 调用了非白名单 op: {qn}\n"
            f"  允许: torch.ops.{self.allowed_ns}.* + 内存调整方法 (contiguous/to/view/...)\n"
            f"  应在 custom kernel 中实现计算, 不允许在 wrapper 中调用 PyTorch op"
        )

    def _is_allowed(self, func: Any) -> bool:
        # 1. torch.ops.<allowed_ns>.*
        if _is_allowed_op_call(func, self.allowed_ns):
            return True
        # 2. Tensor 内存调整方法
        if _is_allowed_tensor_method(func):
            return True
        # 3. torch.* 非计算函数
        if _is_allowed_torch_func(func):
            return True
        # 4. 用户传入的额外白名单 (按 __name__ 比较)
        name = getattr(func, '__name__', '') or ''
        if name in self.extra_allowed:
            return True
        return False


# ---------------------------------------------------------------------------
# Aten 层拦截 (兜底, 防 torch._C._nn.* 这种绕过 Python 层)
# ---------------------------------------------------------------------------

# aten 层允许的 op (内存形态相关, 不参与计算)
_ATEN_ALLOWED: Set[str] = {
    'aten::contiguous', 'aten::_to_copy', 'aten::view', 'aten::reshape',
    'aten::clone', 'aten::detach', 'aten::detach_',
    'aten::unsqueeze', 'aten::squeeze', 'aten::expand', 'aten::expand_as',
    'aten::as_strided', 'aten::flatten', 'aten::t', 'aten::transpose',
    'aten::permute',
    'aten::lift_fresh', 'aten::lift_fresh_copy',
    # 元信息读取
    'aten::size', 'aten::dim', 'aten::numel', 'aten::is_contiguous',
    'aten::stride', 'aten::storage_offset', 'aten::data_ptr',
    'aten::is_floating_point', 'aten::is_complex',
    'aten::sym_size', 'aten::sym_numel', 'aten::sym_stride',
    'aten::resolve_conj', 'aten::resolve_neg', 'aten::alias',
}


class AtenAuditMode(TorchDispatchMode):
    """拦截 aten 层 op, 兜底防 torch._C._nn.* / torch._C._VariableFunctions.* 等绕过。

    用法 (与 OpAuditMode 叠加):
        with OpAuditMode(allowed_ns="mkb"), AtenAuditMode(allowed_ns="mkb"):
            output = submission_func(**kwargs)
    """

    def __init__(self, allowed_ns: str = "mkb",
                 extra_allowed: Optional[Set[str]] = None):
        super().__init__()
        self.allowed_ns = allowed_ns
        self.extra_allowed = set(extra_allowed or ())

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func)  # 形如 'aten::matmul' 或 'mkb::my_op'

        if self._is_allowed(op_name):
            return func(*args, **kwargs)

        raise HackDetected(
            f"[L3-TorchDispatch] submission 触发了非白名单 aten op: {op_name}\n"
            f"  允许: {self.allowed_ns}::* + 内存调整 op\n"
            f"  应在 custom kernel 内实现, 不允许通过 PyTorch dispatcher 走官方实现"
        )

    def _is_allowed(self, op_name: str) -> bool:
        # 1. 自定义 op
        if op_name.startswith(f"{self.allowed_ns}::"):
            return True
        # 2. aten 内存调整白名单
        if op_name in _ATEN_ALLOWED:
            return True
        # prim:: 系列 (PyTorch 内部低层操作)
        if op_name.startswith("prim::"):
            return True
        # 3. 额外白名单
        if op_name in self.extra_allowed:
            return True
        return False
