"""AST 静态黑名单 — L2 防 hack 检查。

入口: audit_submission(source_code, function_name, allowed_op_ns="mkb") -> List[Violation]

扫描 submission 中名为 function_name 的 wrapper 函数体, 命中黑名单 → 返回违规列表。
返回空列表表示通过。

设计取舍:
- 只扫指定 function_name 的函数体 (snake_case op 名), 不扫整个文件
  → 文件顶部允许 import torch / 定义辅助常量
- 函数体内严禁:
    1. torch.nn.* 模块实例化           (nn.Linear / nn.Conv2d / ...)
    2. torch.nn.functional.* 调用      (F.conv2d / F.gelu / ...)
    3. torch.<计算函数>                (matmul / conv / softmax / sum / mean / ...)
    4. Tensor 上的计算方法              (x.matmul / x.softmax / x.sum / ...)
    5. Tensor 间运算符                 (x+y / x*y / x@y)
    6. if/elif/IfExp                  (防 fallback hack)
    7. 函数体内 import / exec / eval  (防动态绕过)
- 函数体内允许:
    1. torch.ops.<allowed_ns>.*       (调用注册的 custom op)
    2. Tensor 内存形态调整: .contiguous() / .to() / .view / .reshape / .clone
    3. 纯 Python 整数算术 / tuple/list/dict 构造
    4. for 循环 (有些 wrapper 需要遍历)

未覆盖的层 (留给 L3 / L4):
- 动态名字 getattr(torch, "matmul")   → L3 dispatch hook 拦截
- 别名链 / helper 模块 / exec          → L3 兜底
- C++ 直接 launch ACLNN              → L4 profiler 黑名单
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# 黑名单定义
# ---------------------------------------------------------------------------

# torch.<这些>(...)  禁止调用 (计算类 op)
TORCH_FORBIDDEN_FUNCS: Set[str] = {
    # 矩阵乘
    'matmul', 'mm', 'bmm', 'addmm', 'addbmm', 'baddbmm', 'einsum',
    # 卷积/池化 (CNN, 但保险也禁)
    'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
    'avg_pool1d', 'avg_pool2d', 'avg_pool3d', 'max_pool1d', 'max_pool2d', 'max_pool3d',
    # 激活
    'relu', 'sigmoid', 'tanh', 'gelu', 'silu', 'softmax', 'log_softmax',
    'leaky_relu', 'elu', 'selu', 'mish', 'hardswish', 'hardsigmoid', 'hardtanh',
    'softplus', 'logsigmoid',
    # 归一化
    'layer_norm', 'batch_norm', 'group_norm', 'instance_norm', 'rms_norm',
    'normalize',
    # 归约
    'sum', 'mean', 'max', 'min', 'prod', 'std', 'var', 'norm',
    'logsumexp', 'argmax', 'argmin', 'cumsum', 'cumprod',
    # 其他常见计算
    'add', 'sub', 'mul', 'div', 'pow', 'exp', 'log', 'sqrt', 'rsqrt',
    'abs', 'neg', 'sign', 'clamp', 'clip',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'where', 'masked_fill', 'masked_select', 'gather', 'scatter',
    'topk', 'sort', 'unique',
    'cat', 'concat', 'stack', 'split', 'chunk',
    # 随机 (避免 hack 用随机绕过)
    'randn', 'rand', 'randint', 'randperm',
    # dropout 也算计算
    'dropout',
}

# Tensor 上的方法名禁止 (x.matmul / x.softmax / x.sum / ...)
TENSOR_FORBIDDEN_METHODS: Set[str] = TORCH_FORBIDDEN_FUNCS | {
    # Tensor 特有的 inplace / 算术变体
    'matmul_', 'add_', 'sub_', 'mul_', 'div_', 'pow_',
    'sum_to_size', 'amax', 'amin',
    'transpose', 'permute',  # transpose 算计算性的, view/reshape 才允许
    'softmax_',
}

# 允许的 Tensor 方法 (内存形态调整, 不参与计算)
TENSOR_ALLOWED_METHODS: Set[str] = {
    'contiguous', 'to', 'view', 'reshape', 'clone', 'detach',
    'unsqueeze', 'squeeze', 'expand', 'expand_as',
    'size', 'shape', 'dim', 'ndim', 'dtype', 'device',
    'is_contiguous', 'is_cuda', 'numel',
    'as_strided', 'flatten', 't', 'mT',
    'data_ptr', 'storage_offset', 'stride',
}

# 完全禁止的 builtin (动态绕过手段)
FORBIDDEN_BUILTINS: Set[str] = {
    'exec', 'eval', 'compile', '__import__', 'getattr', 'setattr',
    'globals', 'locals', 'vars',
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    rule: str            # 命中的规则简称
    detail: str          # 具体说明
    lineno: int          # 行号 (相对 submission 源码)
    col: int = 0
    code_snippet: str = ""

    def __str__(self) -> str:
        loc = f"line {self.lineno}"
        if self.code_snippet:
            return f"[{self.rule}] {loc}: {self.detail}\n    >>> {self.code_snippet}"
        return f"[{self.rule}] {loc}: {self.detail}"


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

class _AstAuditor(ast.NodeVisitor):
    """扫描函数体, 收集违规。"""

    def __init__(self, allowed_op_ns: str, source_lines: List[str]):
        self.allowed_op_ns = allowed_op_ns         # "mkb" 等
        self.source_lines = source_lines
        self.violations: List[Violation] = []

    # ---- 工具 ----

    def _snippet(self, node: ast.AST) -> str:
        if not hasattr(node, 'lineno'):
            return ""
        idx = node.lineno - 1
        if 0 <= idx < len(self.source_lines):
            return self.source_lines[idx].strip()
        return ""

    def _add(self, rule: str, detail: str, node: ast.AST):
        self.violations.append(Violation(
            rule=rule,
            detail=detail,
            lineno=getattr(node, 'lineno', 0),
            col=getattr(node, 'col_offset', 0),
            code_snippet=self._snippet(node),
        ))

    @staticmethod
    def _attr_chain(node: ast.AST) -> Optional[List[str]]:
        """把 a.b.c.d 解析为 ['a', 'b', 'c', 'd']; 解析失败返回 None"""
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return list(reversed(parts))
        return None

    # ---- 调用检查 ----

    def visit_Call(self, node: ast.Call):
        # 优先处理 Name (foo()): 抓 exec/eval/getattr 等动态绕过
        if isinstance(node.func, ast.Name):
            self._check_name_call(node.func.id, node)
        elif isinstance(node.func, ast.Attribute):
            chain = self._attr_chain(node.func)
            if chain is not None:
                self._check_chain_call(chain, node)
        # 其他形态 (lambda 调用 / subscript 调用) 也潜在风险, 但少见
        self.generic_visit(node)

    def _check_chain_call(self, chain: List[str], node: ast.Call):
        """处理 a.b.c(...) 形式调用"""
        head = chain[0]

        # 1. torch.ops.<ns>.<op>(...)  → 白名单, 必须是 allowed_op_ns
        if len(chain) >= 4 and chain[0] == 'torch' and chain[1] == 'ops':
            ns = chain[2]
            if ns != self.allowed_op_ns:
                self._add('forbidden_op_ns',
                          f"torch.ops.{ns}.* 未在白名单 (允许的 ns: {self.allowed_op_ns})",
                          node)
            return  # 合法

        # 2. torch.nn.functional.<x>(...)  → 全禁
        if len(chain) >= 3 and chain[0] == 'torch' and chain[1] == 'nn' and chain[2] == 'functional':
            self._add('torch_nn_functional',
                      f"调用 torch.nn.functional.{chain[3] if len(chain) > 3 else '*'} 被禁止",
                      node)
            return
        if head in ('F', 'nn_functional') and len(chain) >= 2:
            # `from torch.nn import functional as F` 后 F.gelu(...)
            self._add('torch_nn_functional',
                      f"调用 {'.'.join(chain)} (疑似 torch.nn.functional 别名)",
                      node)
            return

        # 3. torch.nn.<X>(...)  → nn.Module 实例化, 全禁
        if len(chain) >= 3 and chain[0] == 'torch' and chain[1] == 'nn':
            self._add('torch_nn_module',
                      f"实例化 torch.nn.{'.'.join(chain[2:])} 被禁止 (不允许 Module 注入)",
                      node)
            return
        if head == 'nn' and len(chain) >= 2:
            self._add('torch_nn_module',
                      f"实例化 nn.{'.'.join(chain[1:])} 被禁止 (不允许 Module 注入)",
                      node)
            return

        # 4. torch.<计算函数>(...)
        if head == 'torch' and len(chain) == 2:
            fname = chain[1]
            if fname in TORCH_FORBIDDEN_FUNCS:
                self._add('torch_compute_call',
                          f"调用 torch.{fname} 被禁止 (计算应在 custom kernel 中完成)",
                          node)
                return

        # 5. <obj>.<method>(...) — Tensor 方法检查
        if len(chain) >= 2:
            method = chain[-1]
            if method in TENSOR_FORBIDDEN_METHODS and method not in TENSOR_ALLOWED_METHODS:
                self._add('tensor_compute_method',
                          f"调用 {'.'.join(chain)} 被禁止 (Tensor 计算方法)",
                          node)
                return

    def _check_name_call(self, name: str, node: ast.Call):
        """处理 foo(...) 形式直接调用"""
        if name in FORBIDDEN_BUILTINS:
            self._add('forbidden_builtin',
                      f"调用 {name}() 被禁止 (动态绕过手段)",
                      node)

    # ---- 运算符检查 ----

    def visit_BinOp(self, node: ast.BinOp):
        # Tensor 间算术运算符: x + y, x * y, x @ y, ...
        # 实际无法在 AST 层面区分 Tensor + Tensor vs int + int
        # 策略: 警告 @ 算子 (这个一定是 matmul); + - * / 太通用, 只警告
        if isinstance(node.op, ast.MatMult):
            self._add('tensor_matmul_operator',
                      f"使用 @ 算子 (matmul) 被禁止",
                      node)
        # 其他算术运算符不强禁 (太多 false positive: stride*2 这种 Python int)
        # 由 L3 运行时拦截兜底
        self.generic_visit(node)

    # ---- 分支检查 ----

    def visit_If(self, node: ast.If):
        self._add('if_branch',
                  "wrapper 函数体内禁止 if/elif 分支 (防 fallback hack)",
                  node)
        # 仍然递归进入, 抓内部其他违规
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp):
        self._add('if_branch',
                  "wrapper 函数体内禁止 三元 if 表达式 (防 fallback hack)",
                  node)
        self.generic_visit(node)

    # ---- 动态绕过 ----

    def visit_Import(self, node: ast.Import):
        names = ", ".join(a.name for a in node.names)
        self._add('inline_import',
                  f"函数体内禁止 import ({names})",
                  node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        names = ", ".join(a.name for a in node.names)
        self._add('inline_import',
                  f"函数体内禁止 from ... import ({node.module}: {names})",
                  node)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def audit_submission(source: str,
                     function_name: str,
                     allowed_op_ns: str = "mkb") -> List[Violation]:
    """扫描 source 中名为 function_name 的函数体, 返回违规列表。

    Args:
        source:        submission 源码字符串
        function_name: 要扫描的 wrapper 函数名 (snake_case 算子名)
        allowed_op_ns: 唯一允许的 torch.ops.<ns> 命名空间, 默认 'mkb'

    Returns:
        Violation 列表, 空表示通过

    Raises:
        SyntaxError: 源码本身语法错
        LookupError: 找不到指定函数
    """
    tree = ast.parse(source)
    target_func = _find_function(tree, function_name)
    if target_func is None:
        raise LookupError(f"submission 中找不到名为 {function_name!r} 的函数")

    source_lines = source.splitlines()
    auditor = _AstAuditor(allowed_op_ns=allowed_op_ns, source_lines=source_lines)
    # 只扫函数体, 不扫整个 module
    for stmt in target_func.body:
        auditor.visit(stmt)
    return auditor.violations


def _find_function(tree: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


# ---------------------------------------------------------------------------
# audit_modelnew — 适配 MultiKernelBench 现有 ModelNew 形态
# ---------------------------------------------------------------------------
# 在保留 ModelNew + class 接口、不改 backend 的前提下,
# 通过严格约束 ModelNew 的结构,达到与 schema 函数等价的防 hack 强度。
#
# __init__ 只允许:
#     - super().__init__()
#     - self.<name> = (torch.)?nn.Parameter(...)        ← 显式声明权重
#     - self.register_buffer(<name>, ...)              ← 显式声明 buffer
# 严禁:
#     - 实例化 nn.<除 Parameter 外的子类>(...)         ← 关掉"塞 nn.Module"的 attack
#
# forward 必须是单行 return:
#     def forward(self, x, ...):
#         return custom_ops_lib.<op>_custom(<args>)
# args 只能是: forward 入参 / self.<name>, 可后接 .contiguous/.to/.view/.reshape/.unsqueeze/.squeeze
# 严禁: 中间赋值 / if-else / for / Tensor 算术 / 任何其他函数调用


# 以下属于 nn 模块的"非计算"成员, 在 __init__ 里允许出现
# (其他 nn.<X> 都视为 Module 实例化, 禁)
_NN_NON_MODULE_NAMES: Set[str] = {'Parameter', 'init', 'utils'}

# forward args 中允许的方法链
_FORWARD_ARG_ALLOWED_METHODS: Set[str] = {
    'contiguous', 'to', 'view', 'reshape', 'unsqueeze', 'squeeze',
    'flatten', 'detach', 'clone',
}


def _find_class(tree: ast.Module, name: str) -> Optional[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_method(cls: ast.ClassDef, name: str) -> Optional[ast.FunctionDef]:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _attr_chain_static(node: ast.AST) -> Optional[List[str]]:
    """模块级 helper, 等价于 _AstAuditor._attr_chain"""
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return list(reversed(parts))
    return None


def _is_super_init_call(node: ast.AST) -> bool:
    """super().__init__() 形态"""
    if not isinstance(node, ast.Call):
        return False
    # node.func 应为 Attribute(value=Call(super), attr='__init__')
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != '__init__':
        return False
    inner = node.func.value
    if not isinstance(inner, ast.Call):
        return False
    if isinstance(inner.func, ast.Name) and inner.func.id == 'super':
        return True
    return False


def _is_self_attr_target(node: ast.AST) -> bool:
    """self.<name> 赋值 target"""
    return (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == 'self')


def _is_nn_parameter_call(node: ast.AST) -> bool:
    """nn.Parameter(...) / torch.nn.Parameter(...) / Parameter(...) 调用"""
    if not isinstance(node, ast.Call):
        return False
    chain = _attr_chain_static(node.func)
    if chain is None:
        return False
    # 接受: ['nn', 'Parameter']  ['torch', 'nn', 'Parameter']  ['Parameter']
    return chain[-1] == 'Parameter' and (
        chain == ['Parameter']
        or chain[-2:] == ['nn', 'Parameter']
    )


def _is_self_register_buffer_call(node: ast.AST) -> bool:
    """self.register_buffer(...) 调用"""
    if not isinstance(node, ast.Call):
        return False
    chain = _attr_chain_static(node.func)
    return chain == ['self', 'register_buffer']


def _is_simple_scalar_value(node: ast.AST) -> bool:
    """判断右值是否是"标量 attr"形态: 字面量 / 入参直接传 / tuple/list of 上面 / 简单算术。

    允许的形态:
        self.eps = 1e-5                  ← Constant
        self.multiplier = multiplier     ← Name (init 入参)
        self.shape = (1, 2)              ← Tuple of constants
        self.epsilon = -1.0              ← UnaryOp(USub, Constant)
        self.scale = 1 / d_model         ← BinOp on simple operands
    禁止 (确保不是绕道引入 tensor / 计算):
        self.x = torch.randn(...)        ← Call
        self.y = nn.Linear(...)          ← Call (已被 nn.Module 检测拦)
        self.z = some.attr.chain         ← 长 Attribute 链 (避开 self.gemm.weight 这种)
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_simple_scalar_value(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_simple_scalar_value(node.left) and _is_simple_scalar_value(node.right)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_simple_scalar_value(e) for e in node.elts)
    return False


def _is_nn_module_instantiation(node: ast.AST) -> bool:
    """nn.<X>(...) / torch.nn.<X>(...) 形态, 且 X 不在白名单 (Parameter/init/utils)"""
    if not isinstance(node, ast.Call):
        return False
    chain = _attr_chain_static(node.func)
    if chain is None:
        return False
    # nn.<X>(...)
    if len(chain) >= 2 and chain[0] == 'nn':
        return chain[1] not in _NN_NON_MODULE_NAMES
    # torch.nn.<X>(...)
    if len(chain) >= 3 and chain[0] == 'torch' and chain[1] == 'nn':
        return chain[2] not in _NN_NON_MODULE_NAMES
    return False


def _check_modelnew_init(method: ast.FunctionDef,
                         source_lines: List[str]) -> List[Violation]:
    """检查 ModelNew.__init__ 内每条语句的形态"""
    violations: List[Violation] = []

    def snippet(node):
        idx = getattr(node, 'lineno', 1) - 1
        return source_lines[idx].strip() if 0 <= idx < len(source_lines) else ""

    def add(rule, detail, node):
        violations.append(Violation(
            rule=rule, detail=detail,
            lineno=getattr(node, 'lineno', 0),
            col=getattr(node, 'col_offset', 0),
            code_snippet=snippet(node),
        ))

    for stmt in method.body:
        # 跳过 docstring
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            continue

        # ① super().__init__()
        if isinstance(stmt, ast.Expr) and _is_super_init_call(stmt.value):
            continue

        # ② self.register_buffer(...)
        if isinstance(stmt, ast.Expr) and _is_self_register_buffer_call(stmt.value):
            continue

        # ③ self.<name> = ...
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and _is_self_attr_target(stmt.targets[0]):

            value = stmt.value
            # ③.1 self.<name> = nn.Parameter(...)  — 学习权重
            if _is_nn_parameter_call(value):
                # 进一步检查 Parameter 内部不能塞 nn.Module 实例化
                for sub in ast.walk(value):
                    if _is_nn_module_instantiation(sub):
                        chain = _attr_chain_static(sub.func)
                        add('init_nn_module',
                            f"__init__ 中通过 nn.Parameter 包装的表达式仍含 nn.Module 实例化: "
                            f"{'.'.join(chain)}",
                            sub)
                continue

            # ③.2 self.<name> = <init 入参名 / 字面量 / 简单运算>  — 标量 attr (multiplier, eps, ...)
            if _is_simple_scalar_value(value):
                continue

            # ③.3 不是 Parameter, 检查是否 nn.Module 实例化
            if _is_nn_module_instantiation(value):
                chain = _attr_chain_static(value.func)
                add('init_nn_module',
                    f"__init__ 禁止实例化 nn.Module 子类: {'.'.join(chain)}(...) "
                    f"— 只允许 nn.Parameter / register_buffer",
                    stmt)
                continue

            # ③.4 其他 self.<name> = <表达式> 都禁 (防任意状态注入)
            add('init_unknown_assign',
                f"__init__ 中 self.{stmt.targets[0].attr} 赋值不允许此形态: "
                f"只允许 nn.Parameter / register_buffer / 标量 attr "
                f"(int/float/bool/str/tuple/list/init 入参)",
                stmt)
            continue

        # ④ 其他都禁
        add('init_unknown_stmt',
            f"__init__ 内只允许 super().__init__() / self.<name> = nn.Parameter(...) / "
            f"self.register_buffer(...), 出现了其他语句 ({type(stmt).__name__})",
            stmt)

    return violations


def _is_allowed_forward_arg(node: ast.AST,
                            forward_input_names: Set[str]) -> Optional[str]:
    """检查 forward 调用的实参是否合法。

    允许:
    - Name(<forward 入参名>)            x
    - Attribute(Name('self'), <name>)  self.weight
    - 上面任一 + .contiguous() / .to(...) / .view(...) / .reshape(...) 等链

    返回错误描述 (None 表示通过)
    """
    cur = node
    # 剥掉允许的方法链
    while isinstance(cur, ast.Call):
        if not isinstance(cur.func, ast.Attribute):
            return f"参数中调用了未知函数 (期望 .contiguous/.to/.view/.reshape 等)"
        method = cur.func.attr
        if method not in _FORWARD_ARG_ALLOWED_METHODS:
            return (f"参数中调用了不允许的方法 .{method}() "
                    f"(只允许 {sorted(_FORWARD_ARG_ALLOWED_METHODS)})")
        cur = cur.func.value

    # 剥掉属性访问 (Name) — 如 self.weight.shape 不允许 (没意义),
    # 但 self.weight 是 Attribute(Name('self'), 'weight') 应允许
    if isinstance(cur, ast.Attribute):
        # 必须是 self.<name>, 且 <name> 不再嵌套 (即 self.weight, 不是 self.x.y)
        if (isinstance(cur.value, ast.Name) and cur.value.id == 'self'):
            return None
        # 拿到嵌套链做人话报错: self.gemm.weight → "self.gemm.weight"
        try:
            chain = _attr_chain_static(cur)
            chain_str = '.'.join(chain) if chain else '<复杂表达式>'
        except Exception:
            chain_str = '<复杂表达式>'
        return f"参数访问了嵌套属性 {chain_str}, 只允许 self.<name> (在 __init__ 中声明的)"

    if isinstance(cur, ast.Name):
        if cur.id in forward_input_names:
            return None
        return f"参数引用了未在 forward 入参中声明的名字: {cur.id}"

    # 常量/None 等也允许 (有时 schema 有 default)
    if isinstance(cur, ast.Constant):
        return None

    return f"参数形态不允许 ({type(cur).__name__})"


def _check_modelnew_forward(method: ast.FunctionDef,
                            source_lines: List[str],
                            op_name: str,
                            allowed_op_module: str) -> List[Violation]:
    """检查 ModelNew.forward 必须是单行 return custom_ops_lib.<op>_custom(...)"""
    violations: List[Violation] = []
    expected_call = f"{op_name}_custom"

    def snippet(node):
        idx = getattr(node, 'lineno', 1) - 1
        return source_lines[idx].strip() if 0 <= idx < len(source_lines) else ""

    def add(rule, detail, node):
        violations.append(Violation(
            rule=rule, detail=detail,
            lineno=getattr(node, 'lineno', 0),
            col=getattr(node, 'col_offset', 0),
            code_snippet=snippet(node),
        ))

    # forward 入参 (除 self)
    forward_input_names: Set[str] = set()
    for p in method.args.args:
        if p.arg != 'self':
            forward_input_names.add(p.arg)

    # 跳过 docstring
    body = list(method.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]

    # 必须只有一条 return 语句
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        add('forward_not_single_return',
            f"forward 必须是单行 return custom_ops_lib.{expected_call}(...) "
            f"(实际有 {len(body)} 条语句)",
            method)
        return violations

    ret = body[0]
    if not isinstance(ret.value, ast.Call):
        add('forward_not_call',
            f"forward 的 return 必须是 custom_ops_lib.{expected_call}(...) 调用",
            ret)
        return violations

    call = ret.value
    # 检查 func 链: <allowed_op_module>.<op>_custom
    chain = _attr_chain_static(call.func)
    if chain is None or len(chain) < 2 or chain[0] != allowed_op_module:
        add('forward_wrong_target',
            f"forward 必须调用 {allowed_op_module}.{expected_call}(...), "
            f"实际调用了 {'.'.join(chain) if chain else '?'}",
            call)
        return violations

    if chain[-1] != expected_call:
        add('forward_wrong_op',
            f"forward 调用的 op 名应为 {expected_call}, 实际是 {chain[-1]}",
            call)
        # 不直接 return, 继续检查 args

    # 检查每个实参
    for i, arg in enumerate(call.args):
        err = _is_allowed_forward_arg(arg, forward_input_names)
        if err:
            add('forward_arg_invalid',
                f"forward 第 {i+1} 个参数不合法: {err}",
                arg)

    # 不允许 kwargs (kernel API 一般不用 kwarg)
    if call.keywords:
        for kw in call.keywords:
            add('forward_unexpected_kwarg',
                f"forward 调用不允许使用关键字参数 {kw.arg}=...",
                kw.value)

    return violations


def audit_modelnew(source: str,
                   op_name: str,
                   allowed_op_module: str = "custom_ops_lib") -> List[Violation]:
    """扫 source 中的 class ModelNew, 检查其 __init__ 和 forward 是否合规。

    Args:
        source:           model_src 字符串
        op_name:          算子名 (snake_case), 如 'gemm_add_relu'
        allowed_op_module: forward 中允许调用的模块名, 默认 'custom_ops_lib'

    Returns:
        Violation 列表, 空表示通过
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [Violation(rule='syntax_error', detail=str(e), lineno=e.lineno or 0)]

    cls = _find_class(tree, 'ModelNew')
    if cls is None:
        return [Violation(rule='missing_class',
                          detail="model_src 中找不到 class ModelNew", lineno=0)]

    init_method = _find_method(cls, '__init__')
    forward_method = _find_method(cls, 'forward')

    source_lines = source.splitlines()
    violations: List[Violation] = []

    if init_method is None:
        violations.append(Violation(
            rule='missing_init',
            detail="ModelNew 缺少 __init__", lineno=cls.lineno))
    else:
        violations.extend(_check_modelnew_init(init_method, source_lines))

    if forward_method is None:
        violations.append(Violation(
            rule='missing_forward',
            detail="ModelNew 缺少 forward", lineno=cls.lineno))
    else:
        violations.extend(_check_modelnew_forward(
            forward_method, source_lines, op_name, allowed_op_module))

    return violations


def format_violations(violations: List[Violation]) -> str:
    """把违规列表格式化为可读的失败信息"""
    if not violations:
        return "OK (no violations)"
    lines = [f"AST 检测到 {len(violations)} 处违规:"]
    for i, v in enumerate(violations, 1):
        lines.append(f"  {i}. {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / 调试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """手动验证:
        python utils/ast_audit.py <source_file> <function_name> [<allowed_op_ns>]
    """
    import sys
    if len(sys.argv) < 3:
        print("usage: ast_audit.py <source_file> <function_name> [<allowed_op_ns>]",
              file=sys.stderr)
        sys.exit(1)
    src_path = sys.argv[1]
    fn_name = sys.argv[2]
    ns = sys.argv[3] if len(sys.argv) > 3 else "mkb"
    with open(src_path) as f:
        src = f.read()
    vios = audit_submission(src, fn_name, allowed_op_ns=ns)
    print(format_violations(vios))
    sys.exit(0 if not vios else 1)
