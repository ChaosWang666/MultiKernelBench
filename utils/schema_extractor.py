"""Schema extractor.

从 reference/<cat>/<op>.py 的 `class Model` + `get_inputs` + `get_init_inputs`
自动推导算子 schema,供 backend 调用、prompt 构造、L2/L3 防护使用。

数据流:
    reference Model class
        |
        v
    extract_schema(ref_path)
        |
        v
    SchemaInfo (init_attrs / parameters / buffers / inputs / outputs)
        |
        +--> backend 调用 build_call_kwargs() 平铺参数,调 submission
        +--> prompt 生成器读 call_signature 嵌入到模板
        +--> L2/L3 用 call_order/op_name 构造白名单
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

# 预先触发 torch._dynamo 加载,避开 PyTorch 2.7.x 的循环导入 bug:
# 某些 op (torch.triu / torch.arange 等) 在 meta 设备上首次调用会触发 _dynamo 加载,
# 而 _dynamo 加载在某些 import 路径下会循环死锁。提前一次性加载解决。
try:
    import torch._dynamo  # noqa: F401
except Exception:
    pass


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SchemaParam:
    """schema 中的单个参数"""
    name: str                                # sanitize 后的参数名 (作为 kwarg)
    kind: str                                # 'init_attr' | 'parameter' | 'buffer' | 'input' | 'output'

    # tensor 类参数 (parameter / buffer / input / output)
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None              # 'float32' / 'float16' / 'int64' / ...

    # init_attr 类参数
    py_type: Optional[str] = None            # 'int' / 'float' / 'str' / 'tuple' / 'NoneType' / ...
    default: Any = None
    has_default: bool = False

    # parameter / buffer 的原始名 (含 '.')
    raw_name: Optional[str] = None


@dataclass
class SchemaInfo:
    """单个算子的完整 schema"""
    op_name: str                                            # 算子名 (snake_case)
    category: str                                           # 类别
    init_attrs:  List[SchemaParam] = field(default_factory=list)
    parameters:  List[SchemaParam] = field(default_factory=list)
    buffers:     List[SchemaParam] = field(default_factory=list)
    inputs:      List[SchemaParam] = field(default_factory=list)
    outputs:     List[SchemaParam] = field(default_factory=list)

    # 派生
    call_order:      List[str] = field(default_factory=list)   # submission 函数参数顺序
    call_signature:  str = ""                                  # 给 prompt 用的可读签名

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [f"=== {self.op_name} ({self.category}) ===",
                 f"signature: {self.call_signature}",
                 f"call_order: {self.call_order}"]
        if self.init_attrs:
            lines.append(f"init_attrs ({len(self.init_attrs)}):")
            for p in self.init_attrs:
                d = f" = {p.default!r}" if p.has_default else ""
                lines.append(f"    {p.name}: {p.py_type}{d}")
        if self.parameters:
            lines.append(f"parameters ({len(self.parameters)}):")
            for p in self.parameters:
                lines.append(f"    {p.name}: {p.dtype}{list(p.shape)}  (raw={p.raw_name})")
        if self.buffers:
            lines.append(f"buffers ({len(self.buffers)}):")
            for p in self.buffers:
                lines.append(f"    {p.name}: {p.dtype}{list(p.shape)}  (raw={p.raw_name})")
        if self.inputs:
            lines.append(f"inputs ({len(self.inputs)}):")
            for p in self.inputs:
                lines.append(f"    {p.name}: {p.dtype}{list(p.shape)}")
        if self.outputs:
            lines.append(f"outputs ({len(self.outputs)}):")
            for p in self.outputs:
                lines.append(f"    {p.name}: {p.dtype}{list(p.shape)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """'ln.weight' -> 'ln_weight', 'mha.in_proj_weight' -> 'mha_in_proj_weight'"""
    return name.replace('.', '_')


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).replace('torch.', '')


def _py_type_str(value: Any) -> str:
    return type(value).__name__


def _load_reference_module(ref_path: str, mod_name: str = "ref_op_module"):
    """动态加载 reference/<cat>/<op>.py 为独立 module,避免污染 sys.modules"""
    spec = importlib.util.spec_from_file_location(mod_name, ref_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 reference 文件: {ref_path}")
    module = importlib.util.module_from_spec(spec)
    # 不写入 sys.modules,避免多算子互相覆盖
    spec.loader.exec_module(module)
    return module


class _MetaFactoryContext:
    """让 torch 的 factory function 默认在 meta 设备上创建 tensor。

    meta tensor 只有 shape/dtype 元数据,不分配真实内存。
    schema 提取只需要形状信息,无需真实 tensor 内容,所以这是零开销 schema 推导。

    覆盖范围:
    - torch.rand / randn / empty / zeros / ones / full / arange / eye / tensor
    - 同名的 _like 系列
    Reference 用户代码里 `torch.rand(...)` 不显式指定 device 时,会被 patch 成 meta。
    nn.Linear / nn.LayerNorm 等内部用 torch.empty 创建 weight,也走 meta。

    退出时 restore 所有 patch。
    """
    _FACTORY_FUNCS = (
        'rand', 'randn', 'empty', 'zeros', 'ones', 'full',
        'arange', 'eye', 'tensor',
        'rand_like', 'randn_like', 'empty_like',
        'zeros_like', 'ones_like', 'full_like',
    )

    def __enter__(self):
        self._originals: Dict[str, Any] = {}
        for name in self._FACTORY_FUNCS:
            if not hasattr(torch, name):
                continue
            orig = getattr(torch, name)
            self._originals[name] = orig
            setattr(torch, name, self._wrap(orig))
        return self

    def __exit__(self, *exc):
        for name, orig in self._originals.items():
            setattr(torch, name, orig)
        return False

    @staticmethod
    def _wrap(orig):
        def wrapper(*args, **kwargs):
            kwargs.setdefault('device', 'meta')
            return orig(*args, **kwargs)
        return wrapper


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def extract_schema(ref_path: str,
                   op_name: str,
                   category: str = "",
                   verbose: bool = False) -> SchemaInfo:
    """从 reference Python 文件自动推导 SchemaInfo。

    Args:
        ref_path: reference/<cat>/<op>.py 的绝对路径
        op_name: 算子名 (snake_case)
        category: 类别 (attention / matmul / fuse / ...)
        verbose: 是否打印 forward-on-meta 失败等非致命 warning (默认静默)

    Returns:
        SchemaInfo
    """
    module = _load_reference_module(ref_path)

    if not hasattr(module, 'Model'):
        raise AttributeError(f"{ref_path} 中找不到 class Model")
    if not hasattr(module, 'get_inputs'):
        raise AttributeError(f"{ref_path} 中找不到 get_inputs()")
    if not hasattr(module, 'get_init_inputs'):
        raise AttributeError(f"{ref_path} 中找不到 get_init_inputs()")

    Model = module.Model
    get_inputs = module.get_inputs
    get_init_inputs = module.get_init_inputs

    schema = SchemaInfo(op_name=op_name, category=category)

    # 整个 schema 提取过程都在 meta 设备上,零内存开销
    with _MetaFactoryContext(), torch.no_grad():

        # 1. init_attrs: __init__ 签名 (除 self) + get_init_inputs() 的实际值
        init_sig = inspect.signature(Model.__init__)
        init_params = [
            p for p in init_sig.parameters.values()
            if p.name != 'self' and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        init_values = list(get_init_inputs())

        for i, p in enumerate(init_params):
            val = init_values[i] if i < len(init_values) else None
            py_type = _py_type_str(val) if val is not None else (
                _py_type_str(p.default) if p.default is not inspect.Parameter.empty else None
            )
            has_default = p.default is not inspect.Parameter.empty
            schema.init_attrs.append(SchemaParam(
                name=p.name,
                kind='init_attr',
                py_type=py_type,
                default=p.default if has_default else None,
                has_default=has_default,
            ))

        # 2. 实例化 Model (全部 weight/buffer 在 meta 上)
        model = Model(*init_values)
        model.eval()

        # 3. parameters: nn.Parameter (含 nn.Linear / nn.LayerNorm 等模块内部的)
        for raw_name, param in model.named_parameters():
            schema.parameters.append(SchemaParam(
                name=_sanitize_name(raw_name),
                kind='parameter',
                shape=tuple(param.shape),
                dtype=_dtype_str(param.dtype),
                raw_name=raw_name,
            ))

        # 4. buffers: register_buffer (causal_mask / running_mean 等)
        for raw_name, buf in model.named_buffers():
            schema.buffers.append(SchemaParam(
                name=_sanitize_name(raw_name),
                kind='buffer',
                shape=tuple(buf.shape),
                dtype=_dtype_str(buf.dtype),
                raw_name=raw_name,
            ))

        # 5. inputs: forward 签名 + get_inputs() 实际值 (meta tensor)
        forward_sig = inspect.signature(Model.forward)
        forward_params = [
            p for p in forward_sig.parameters.values()
            if p.name != 'self' and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        sample_inputs = list(get_inputs())

        for i, p in enumerate(forward_params):
            if i >= len(sample_inputs):
                break
            t = sample_inputs[i]
            if isinstance(t, torch.Tensor):
                schema.inputs.append(SchemaParam(
                    name=p.name,
                    kind='input',
                    shape=tuple(t.shape),
                    dtype=_dtype_str(t.dtype),
                ))
            else:
                # 非 tensor 输入 (罕见),按 init_attr 处理
                schema.inputs.append(SchemaParam(
                    name=p.name,
                    kind='input',
                    py_type=_py_type_str(t),
                    default=t,
                    has_default=True,
                ))

        # 6. outputs: 在 meta 上跑一次 forward 拿 shape/dtype
        # 注意: 部分 op 在 meta backend 不支持 (如某些 attention/MHA 内核),
        # 失败时不阻塞 schema 提取,outputs 留空; 不影响 L0/L1/L2/L3 检测,
        # 仅 prompt 嵌入时少一行 Output shape 信息。
        try:
            out = model(*sample_inputs)
            schema.outputs = _flatten_outputs(out)
        except Exception as e:
            if verbose:
                print(f"[schema_extractor] note: {op_name} forward on meta backend "
                      f"failed ({type(e).__name__}); schema.outputs left empty. "
                      f"This is non-fatal — L1/L2/L3 checks unaffected.",
                      file=sys.stderr)

    # 7. 派生 call_order 和 call_signature
    schema.call_order = (
        [p.name for p in schema.inputs]
        + [p.name for p in schema.parameters]
        + [p.name for p in schema.buffers]
        + [p.name for p in schema.init_attrs]
    )
    schema.call_signature = _build_signature(op_name, schema)

    return schema


def _flatten_outputs(out: Any) -> List[SchemaParam]:
    """支持 tuple/list/dict/Tensor 多种返回形态"""
    result: List[SchemaParam] = []

    def _walk(name: str, item: Any):
        if isinstance(item, torch.Tensor):
            result.append(SchemaParam(
                name=name,
                kind='output',
                shape=tuple(item.shape),
                dtype=_dtype_str(item.dtype),
            ))
        elif isinstance(item, (tuple, list)):
            for i, sub in enumerate(item):
                _walk(f"{name}_{i}", sub)
        elif isinstance(item, dict):
            for k, v in item.items():
                _walk(f"{name}_{k}", v)
        # 其他类型 (None / scalar) 忽略

    if isinstance(out, torch.Tensor):
        _walk('y', out)
    elif isinstance(out, (tuple, list)):
        for i, sub in enumerate(out):
            _walk(f'y{i}', sub)
    elif isinstance(out, dict):
        for k, v in out.items():
            _walk(f'y_{k}', v)
    else:
        # 单个 tensor 之外的标量返回值,忽略
        pass

    return result


def _build_signature(op_name: str, schema: SchemaInfo) -> str:
    parts: List[str] = []
    for p in schema.inputs:
        parts.append(f"{p.name}: Tensor")
    for p in schema.parameters:
        parts.append(f"{p.name}: Tensor")
    for p in schema.buffers:
        parts.append(f"{p.name}: Tensor")
    for p in schema.init_attrs:
        ty = p.py_type if p.py_type else 'Any'
        if p.has_default:
            parts.append(f"{p.name}: {ty} = {p.default!r}")
        else:
            parts.append(f"{p.name}: {ty}")
    if len(schema.outputs) <= 1:
        ret = "Tensor"
    else:
        ret = f"Tuple[{', '.join('Tensor' for _ in schema.outputs)}]"
    return f"def {op_name}({', '.join(parts)}) -> {ret}"


# ---------------------------------------------------------------------------
# 给 backend 用: 把 reference Model 的实际值平铺成 submission 函数的 kwargs
# ---------------------------------------------------------------------------

def build_call_kwargs(model: torch.nn.Module,
                      inputs: List[torch.Tensor],
                      init_values: List[Any],
                      schema: SchemaInfo) -> Dict[str, Any]:
    """把 reference 实例化后的状态 + 当前 inputs 平铺成 submission 函数的 kwargs。

    submission 函数声明的签名是 schema.call_signature,
    本函数确保按 schema.call_order 顺序提供所有命名参数。
    """
    kwargs: Dict[str, Any] = {}

    # inputs
    for i, p in enumerate(schema.inputs):
        if i < len(inputs):
            kwargs[p.name] = inputs[i]

    # parameters from model.named_parameters()
    param_dict = dict(model.named_parameters())
    for p in schema.parameters:
        if p.raw_name not in param_dict:
            raise KeyError(f"schema.parameter {p.raw_name} 不在 model.named_parameters() 中")
        kwargs[p.name] = param_dict[p.raw_name]

    # buffers from model.named_buffers()
    buf_dict = dict(model.named_buffers())
    for p in schema.buffers:
        if p.raw_name not in buf_dict:
            raise KeyError(f"schema.buffer {p.raw_name} 不在 model.named_buffers() 中")
        kwargs[p.name] = buf_dict[p.raw_name]

    # init_attrs
    for i, p in enumerate(schema.init_attrs):
        if i < len(init_values):
            kwargs[p.name] = init_values[i]
        elif p.has_default:
            kwargs[p.name] = p.default
        else:
            raise ValueError(f"schema.init_attr {p.name} 既无 default 也无 init_value")

    return kwargs


# ---------------------------------------------------------------------------
# CLI / 调试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """手动验证:
        python utils/schema_extractor.py <ref_path> [op_name] [category] [--json]
    """
    import json as _json
    args = [a for a in sys.argv[1:] if a not in ("--json", "-v", "--verbose")]
    as_json = "--json" in sys.argv
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    if not args:
        print("usage: schema_extractor.py <ref_path> [op_name] [category] [--json] [-v]",
              file=sys.stderr)
        sys.exit(1)
    ref = args[0]
    name = args[1] if len(args) > 1 else ref.rsplit('/', 1)[-1].removesuffix('.py')
    cat = args[2] if len(args) > 2 else ""
    info = extract_schema(ref, name, cat, verbose=verbose)
    if as_json:
        print(_json.dumps(info.to_dict(), default=str))
    else:
        print(info.summary())
