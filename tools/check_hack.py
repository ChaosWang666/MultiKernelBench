"""一把跑 L0+L1+L2+L3 防 hack 检查。

默认**批量模式**: 扫提交目录下所有 `<op_name>.txt`, 用文件名在 reference 树
(`reference/<category>/<op>.py`) 里查 reference, 每算子独立 subprocess 跑检查,
汇总报告。

不依赖 selected_ops.yaml: 用户提交了什么就检测什么; 只要 reference 树里能找到
同名算子, 就跑完整 L0+L1+L2+L3。

用法:
    # 批量 (默认): 扫目录里所有 *.txt
    python tools/check_hack.py <submissions_dir>
    python tools/check_hack.py <submissions_dir> --op multi_query_attention   # 只跑一个

    # 单算子 (内部用 / 调试)
    python tools/check_hack.py --single <reference.py> <submission.txt> [<op_name>]

提交目录文件命名: `<op_name>.txt` (op_name = `reference/*/<op_name>.py` 的 stem)

退出码:
    0   全部通过 (含 SKIP)
    1   有 op 检测失败
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.schema_extractor import (
    extract_schema, SchemaInfo, _MetaFactoryContext, _sanitize_name,
)
from utils.ast_audit import audit_modelnew
from utils.runtime_guard import OpAuditMode, HackDetected


REQUIRED_STRINGS = (
    'project_json_src', 'host_tiling_src', 'host_operator_src',
    'kernel_src', 'python_bind_src', 'model_src',
)

THIS_DIR = Path(__file__).resolve().parent
MKB_ROOT = THIS_DIR.parent
DEFAULT_REF_ROOT = MKB_ROOT / "reference"


# ===========================================================================
# 单算子检查 (核心逻辑, 单算子模式 + 批量模式的子进程都调用它)
# ===========================================================================

@dataclass
class LayerResult:
    """内部使用: 单层检查结果 (L0/L1/L2/L3 各对应一个).

    name 是内部编号(L0/L1/L2/L3), 渲染时不暴露给用户;
    issues 是面向用户的"失败原因"自然语言描述, 用于报告。
    """
    name: str
    status: str
    detail: str = ""               # 单行简短结论 (内部用 / 单算子模式用)
    issues: List[str] = field(default_factory=list)  # 面向用户的失败原因列表

    def is_fail(self) -> bool:
        return self.status == 'FAIL'


@dataclass
class CheckResult:
    op_name: str
    layers: List[LayerResult]
    overall: str        # 'PASS' / 'FAIL' / 'SKIP' / 'ERROR'
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'op_name': self.op_name,
            'overall': self.overall,
            'note': self.note,
            'layers': [{'name': l.name, 'status': l.status,
                        'detail': l.detail, 'issues': list(l.issues)}
                       for l in self.layers],
        }

    @property
    def all_issues(self) -> List[str]:
        """汇总所有失败层的 issue 列表 (面向用户的报告内容)"""
        out: List[str] = []
        for l in self.layers:
            if l.status in ('FAIL', 'WARN'):
                out.extend(l.issues)
        if not out and self.note:
            out.append(self.note)
        return out


def parse_six_string_file(txt_path: str) -> Dict[str, str]:
    code = Path(txt_path).read_text()
    ns: Dict[str, Any] = {}
    exec(compile(code, txt_path, 'exec'), ns)
    missing = [k for k in REQUIRED_STRINGS if k not in ns]
    if missing:
        raise ValueError(f"缺少字符串变量: {missing}")
    out = {k: ns[k] for k in REQUIRED_STRINGS}
    for k, v in out.items():
        if not isinstance(v, str):
            raise TypeError(f"{k} 不是字符串 (实际 {type(v).__name__})")
    return out


def _instantiate_modelnew_on_meta(model_src: str, init_values: List[Any]):
    """在 meta 设备上实例化 ModelNew, 同时 stub 掉 custom_ops_lib 让 forward 可调。"""
    fake_lib = type('FakeCustomOpsLib', (), {})()

    def _make_stub(name):
        def _f(*args, **kwargs):
            for a in args:
                if isinstance(a, torch.Tensor):
                    return a.clone()
            return torch.empty(1, device='meta')
        _f.__name__ = name
        return _f

    class _Lazy:
        def __getattr__(self, name):
            return _make_stub(name)

    sys.modules['custom_ops_lib'] = _Lazy()
    if 'torch_npu' not in sys.modules:
        sys.modules['torch_npu'] = type('m', (), {})()

    ns: Dict[str, Any] = {}
    with _MetaFactoryContext():
        exec(compile(model_src, '<model_src>', 'exec'), ns)

    Model = ns.get('ModelNew')
    if Model is None:
        raise RuntimeError("model_src 中找不到 class ModelNew")

    with _MetaFactoryContext(), torch.no_grad():
        model = Model(*init_values)
        model.eval()
    return model, ns


def _check_l1_schema(schema: SchemaInfo, model: torch.nn.Module) -> List[str]:
    sub_param = {_sanitize_name(n) for n, _ in model.named_parameters()}
    sub_buf = {_sanitize_name(n) for n, _ in model.named_buffers()}
    expected_p = {p.name for p in schema.parameters}
    expected_b = {p.name for p in schema.buffers}

    issues: List[str] = []
    for name in sorted(expected_p - sub_param - sub_buf):
        issues.append(f"missing parameter: {name}")
    for name in sorted(expected_b - sub_param - sub_buf):
        issues.append(f"missing buffer: {name}")
    extra = (sub_param | sub_buf) - expected_p - expected_b
    if extra:
        issues.append(f"WARN: extra param/buffer: {sorted(extra)}")
    return issues


def _build_meta_forward_inputs(schema: SchemaInfo) -> List[torch.Tensor]:
    inputs = []
    for p in schema.inputs:
        if p.shape is not None and p.dtype is not None:
            inputs.append(torch.empty(*p.shape,
                                      dtype=getattr(torch, p.dtype),
                                      device='meta'))
        elif p.has_default:
            inputs.append(p.default)
        else:
            inputs.append(None)
    return inputs


def _load_init_values(ref_path: str) -> List[Any]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_ref_for_init", ref_path)
    mod = importlib.util.module_from_spec(spec)
    with _MetaFactoryContext():
        spec.loader.exec_module(mod)
    return list(mod.get_init_inputs())


def _violation_to_issue(v) -> str:
    """把 audit_modelnew 的 Violation 翻译成面向用户的自然语言"""
    snippet = f"  >>> {v.code_snippet}" if v.code_snippet else ""
    rule_map = {
        'init_nn_module':
            f"line {v.lineno}: __init__ 实例化了 nn.Module 子类 — "
            f"只允许 nn.Parameter / register_buffer{snippet}",
        'init_unknown_assign':
            f"line {v.lineno}: __init__ 中赋值形态不允许, "
            f"只允许 nn.Parameter / register_buffer / 标量 init_attr{snippet}",
        'init_unknown_stmt':
            f"line {v.lineno}: __init__ 包含未知语句类型 ({v.detail}){snippet}",
        'forward_not_single_return':
            f"line {v.lineno}: forward 必须是单行 return custom_ops_lib.<op>_custom(...){snippet}",
        'forward_not_call':
            f"line {v.lineno}: forward 的 return 必须调用 custom_ops_lib.<op>_custom(...){snippet}",
        'forward_wrong_target':
            f"line {v.lineno}: forward 调用了错误的目标 ({v.detail}){snippet}",
        'forward_wrong_op':
            f"line {v.lineno}: forward 调用的 op 名不匹配 ({v.detail}){snippet}",
        'forward_arg_invalid':
            f"line {v.lineno}: {v.detail.split(':', 1)[-1].strip()}{snippet}",
        'forward_unexpected_kwarg':
            f"line {v.lineno}: forward 调用不允许 keyword 参数{snippet}",
        'missing_class':
            "model_src 中找不到 class ModelNew",
        'missing_init':
            "ModelNew 缺少 __init__ 方法",
        'missing_forward':
            "ModelNew 缺少 forward 方法",
        'syntax_error':
            f"model_src 语法错误: {v.detail}",
        'torch_compute_call':
            f"line {v.lineno}: 调用了 PyTorch 计算函数 ({v.detail}){snippet}",
        'torch_nn_functional':
            f"line {v.lineno}: 调用了 torch.nn.functional ({v.detail}){snippet}",
        'torch_nn_module':
            f"line {v.lineno}: 实例化或调用了 nn.Module 子类 ({v.detail}){snippet}",
        'tensor_compute_method':
            f"line {v.lineno}: 调用了 Tensor 计算方法 ({v.detail}){snippet}",
        'tensor_matmul_operator':
            f"line {v.lineno}: 使用了 @ matmul 运算符{snippet}",
        'if_branch':
            f"line {v.lineno}: 函数体内不允许 if/elif 分支 (防 fallback hack){snippet}",
        'inline_import':
            f"line {v.lineno}: 函数体内不允许 import{snippet}",
        'forbidden_builtin':
            f"line {v.lineno}: 调用了被禁的 builtin ({v.detail}){snippet}",
        'forbidden_op_ns':
            f"line {v.lineno}: 调用了未授权的 torch.ops namespace ({v.detail}){snippet}",
    }
    return rule_map.get(v.rule, f"line {v.lineno}: [{v.rule}] {v.detail}{snippet}")


# C++ 黑名单 (kernel_src / python_bind_src)
# 用于抓"在 C++ 里明文调 ATen / ACLNN / 厂商加速 API 走代算"的 hack
# 注: 这是粗糙的字符串匹配, 真正彻底防御要靠 L4 profiler kernel 黑名单
_CPP_BLACKLIST = {
    # ATen C++ API
    r'\bat::matmul\b': 'at::matmul',
    r'\bat::mm\b': 'at::mm',
    r'\bat::bmm\b': 'at::bmm',
    r'\bat::conv1d\b': 'at::conv1d',
    r'\bat::conv2d\b': 'at::conv2d',
    r'\bat::conv3d\b': 'at::conv3d',
    r'\bat::softmax\b': 'at::softmax',
    r'\bat::gelu\b': 'at::gelu',
    r'\bat::relu\b': 'at::relu',
    r'\bat::layer_norm\b': 'at::layer_norm',
    r'\bat::group_norm\b': 'at::group_norm',
    r'\bat::batch_norm\b': 'at::batch_norm',
    r'\bat::native::': 'at::native::*',
    r'\bat::nn::functional': 'at::nn::functional::*',
    # ACLNN 直 launch (非提交者自己的 op)
    r'\baclnnMatmul\b': 'aclnnMatmul',
    r'\baclnnConvolution\b': 'aclnnConvolution',
    r'\baclnnSoftmax\b': 'aclnnSoftmax',
    r'\baclnnGelu\b': 'aclnnGelu',
    r'\baclnnLayerNorm\b': 'aclnnLayerNorm',
    r'\baclnnGroupNorm\b': 'aclnnGroupNorm',
    r'\baclnnBatchNorm\b': 'aclnnBatchNorm',
}


def _audit_cpp_blocks(strings: Dict[str, str]) -> List[str]:
    """扫 kernel_src / python_bind_src, 抓明文 ATen / ACLNN 代算 API。

    粗糙但便宜。彻底防御看 L4 profiler kernel 黑名单。
    """
    import re
    issues: List[str] = []
    for src_name in ('kernel_src', 'python_bind_src'):
        src = strings.get(src_name, '')
        if not src:
            continue
        # 跳过自己注册的 op (例如提交里会出现 aclnnGemmAddReluCustom 这种命名)
        for pattern, label in _CPP_BLACKLIST.items():
            m = re.search(pattern, src)
            if m:
                # 找行号
                line_no = src[:m.start()].count('\n') + 1
                issues.append(
                    f"{src_name} line {line_no}: 命中 C++ 黑名单关键字 `{label}` "
                    f"(在 C++ 里直接调用 PyTorch ATen / ACLNN 官方算子属于代算, 不允许)"
                )
    return issues


def check_one(ref_path: str, txt_path: str, op_name: str,
              verbose: bool = False) -> CheckResult:
    """单算子完整 L0+L1+L2+L3 检查。"""
    layers: List[LayerResult] = []

    # ---- L0: 解析 6 个字符串 ----
    try:
        strings = parse_six_string_file(txt_path)
        layers.append(LayerResult('L0', 'OK', '6 strings present'))
    except Exception as e:
        layers.append(LayerResult('L0', 'FAIL',
                                  detail=f"{type(e).__name__}: {e}",
                                  issues=[f"提交文件解析失败: {e}"]))
        return CheckResult(op_name, layers, 'FAIL')

    # ---- L1: schema 推导 + ModelNew 参数覆盖 ----
    try:
        schema = extract_schema(ref_path, op_name, "")
    except Exception as e:
        msg = f"schema 推导失败: {type(e).__name__}: {e}"
        layers.append(LayerResult('L1', 'FAIL', detail=msg,
                                  issues=[f"reference 算子定义无法解析: {e}"]))
        return CheckResult(op_name, layers, 'FAIL')

    try:
        init_values = _load_init_values(ref_path)
        model, _ = _instantiate_modelnew_on_meta(strings['model_src'], init_values)
    except Exception as e:
        # 截短 traceback (PyTorch 异常常带几页签名候选, 用户看主要错误就够)
        err_short = str(e).splitlines()[0][:200]
        msg = f"ModelNew 实例化失败: {type(e).__name__}: {err_short}"
        issues = [f"ModelNew 实例化失败 (检查 __init__ 签名与 reference 是否一致 / "
                  f"nn.Parameter 的 shape 是否正确): {type(e).__name__}: {err_short}"]
        layers.append(LayerResult('L1', 'FAIL', detail=msg, issues=issues))
        return CheckResult(op_name, layers, 'FAIL')

    l1_issues = _check_l1_schema(schema, model)
    hard_l1 = [i for i in l1_issues if not i.startswith("WARN")]
    if hard_l1:
        # 把内部短语转成自然语言
        issues = []
        for s in hard_l1:
            if s.startswith("missing parameter:"):
                pname = s.split(":", 1)[1].strip()
                issues.append(f"ModelNew 缺少必需的权重参数: {pname}  "
                              f"(reference 中应该声明为 nn.Parameter)")
            elif s.startswith("missing buffer:"):
                pname = s.split(":", 1)[1].strip()
                issues.append(f"ModelNew 缺少必需的 buffer: {pname}  "
                              f"(reference 中应该用 register_buffer 注册)")
            else:
                issues.append(s)
        layers.append(LayerResult('L1', 'FAIL',
                                  detail='; '.join(hard_l1),
                                  issues=issues))
    else:
        layers.append(LayerResult('L1', 'OK',
                                  f"{len(schema.parameters)} param + "
                                  f"{len(schema.buffers)} buffer matched"))

    # ---- L2: AST 黑名单 ----
    try:
        violations = audit_modelnew(strings['model_src'], op_name)
        if violations:
            detail = '; '.join(f"{v.rule}@line{v.lineno}" for v in violations[:5])
            if len(violations) > 5:
                detail += f"  (+{len(violations) - 5} more)"
            issues = [_violation_to_issue(v) for v in violations]
            layers.append(LayerResult('L2', 'FAIL', detail=detail, issues=issues))
        else:
            layers.append(LayerResult('L2', 'OK', 'structure clean'))
    except Exception as e:
        layers.append(LayerResult('L2', 'FAIL',
                                  detail=f"AST 异常: {type(e).__name__}: {e}",
                                  issues=[f"AST 扫描异常: {type(e).__name__}: {e}"]))

    # 同步扫 C++ 部分 (kernel_src / python_bind_src) 的黑名单关键字 — L2 扩展
    cpp_issues = _audit_cpp_blocks(strings)
    if cpp_issues:
        # 把 C++ 违规也归入 L2
        for l in layers:
            if l.name == 'L2':
                if l.status == 'OK':
                    l.status = 'FAIL'
                    l.detail = f"C++ 黑名单命中 ({len(cpp_issues)})"
                else:
                    l.detail += f"; C++ +{len(cpp_issues)}"
                l.issues.extend(cpp_issues)
                break

    # ---- L3: 运行时拦截 ----
    try:
        inputs = _build_meta_forward_inputs(schema)
        with OpAuditMode(allowed_ns="__never_match__"):
            model(*inputs)
        layers.append(LayerResult('L3', 'OK', 'no illegal pytorch call'))
    except HackDetected as e:
        first_line = str(e).splitlines()[0] if str(e) else 'HackDetected'
        issue = f"forward 调用了非白名单 PyTorch 操作: {first_line.replace('[L3-TorchFunction] submission ', '')[:200]}"
        layers.append(LayerResult('L3', 'FAIL', detail=first_line[:200], issues=[issue]))
    except Exception as e:
        # meta backend 不支持的 op 不算 hack
        msg = f"{type(e).__name__}: {str(e)[:100]}"
        layers.append(LayerResult('L3', 'WARN',
                                  detail=f"meta dry-run threw {msg} (not hack)"))

    overall = 'PASS' if not any(l.is_fail() for l in layers) else 'FAIL'
    return CheckResult(op_name, layers, overall)


# ===========================================================================
# 批量模式
# ===========================================================================

def _build_op_index(ref_root: Path) -> Dict[str, Path]:
    """扫 reference 树, 建 op_name → ref_path 映射。

    例: reference/fuse/gemm_add_relu.py → {'gemm_add_relu': '<abspath>/fuse/gemm_add_relu.py'}
    同名算子如果在多个 category 下出现, 后扫到的覆盖前面的 (实际很少出现)。
    """
    index: Dict[str, Path] = {}
    if not ref_root.is_dir():
        return index
    for category_dir in sorted(ref_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for py in sorted(category_dir.glob('*.py')):
            index[py.stem] = py.resolve()
    return index


def _run_subprocess_check(ref_path: Path, sub_path: Path,
                          op_name: str, timeout: int = 180) -> Optional[Dict[str, Any]]:
    """启子进程跑单算子 check, 返回 JSON 解析结果; 失败返回 None"""
    cmd = [sys.executable, str(THIS_DIR / "check_hack.py"),
           "--single", "--json",
           str(ref_path), str(sub_path), op_name]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=str(MKB_ROOT))
    except subprocess.TimeoutExpired:
        return {'op_name': op_name, 'overall': 'FAIL',
                'note': f'subprocess TIMEOUT (>{timeout}s)', 'layers': []}

    # JSON 输出在最后一行 (格式: __RESULT_JSON__:{...})
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("__RESULT_JSON__:"):
            try:
                return json.loads(line[len("__RESULT_JSON__:"):])
            except json.JSONDecodeError:
                pass
    # 没有 JSON, 子进程肯定挂了
    err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else f"exit {p.returncode}"
    return {'op_name': op_name, 'overall': 'ERROR',
            'note': f'subprocess error: {err}', 'layers': []}


def check_batch(submissions_dir: Path,
                ref_root: Optional[Path] = None,
                only_op: Optional[str] = None,
                use_subprocess: bool = True,
                verbose: bool = False) -> int:
    ref_root = (ref_root or DEFAULT_REF_ROOT).resolve()
    if not ref_root.is_dir():
        print(f"FATAL: reference root not found: {ref_root}", file=sys.stderr)
        return 2

    op_index = _build_op_index(ref_root)

    # 扫提交目录所有 .txt
    submissions = sorted(submissions_dir.glob('*.txt'))
    if only_op:
        submissions = [s for s in submissions if s.stem == only_op]
        if not submissions:
            print(f"FATAL: no submission for op {only_op!r} under {submissions_dir}",
                  file=sys.stderr)
            return 2

    if not submissions:
        print(f"WARN: no .txt submissions found under {submissions_dir}",
              file=sys.stderr)
        return 0

    print("=" * 80)
    print(f"  Submissions dir: {submissions_dir}")
    print(f"  Reference root:  {ref_root}")
    print(f"  Submissions:     {len(submissions)}")
    print("=" * 80)
    print(f"\n{'OP':<55s} {'RESULT':<8s}{'ISSUES':<7s}")
    print("-" * 80)

    results: List[Dict[str, Any]] = []
    summary = {'PASS': 0, 'FAIL': 0, 'SKIP': 0, 'ERROR': 0}

    for sub_path in submissions:
        op_name = sub_path.stem
        ref_path = op_index.get(op_name)

        if ref_path is None:
            # 提交了但 reference 树里没有同名算子, 无法做 schema 比对
            print(f"{op_name:<55s} {'SKIP':<8s}{'-':<7s}"
                  f"  (no reference under {ref_root.name}/)")
            results.append({'op_name': op_name, 'overall': 'SKIP',
                            'note': 'no reference found', 'layers': []})
            summary['SKIP'] += 1
            continue

        if use_subprocess:
            res = _run_subprocess_check(ref_path, sub_path, op_name)
        else:
            try:
                cr = check_one(str(ref_path), str(sub_path), op_name, verbose=verbose)
                res = cr.to_dict()
            except Exception as e:
                res = {'op_name': op_name, 'overall': 'ERROR',
                       'note': f'{type(e).__name__}: {e}', 'layers': []}

        results.append(res)
        summary[res['overall']] = summary.get(res['overall'], 0) + 1

        # 渲染单行 — 不暴露 L0/L1/L2/L3
        issues = _collect_issues(res)
        issues_cell = str(len(issues)) if issues else '-'
        print(f"{op_name:<55s} {res['overall']:<8s}{issues_cell:<7s}")

    # ---- 汇总 ----
    print("-" * 80)
    total = len(submissions)
    print("\n===== SUMMARY =====")
    for k in ('PASS', 'FAIL', 'SKIP', 'ERROR'):
        n = summary.get(k, 0)
        if n > 0 or k in ('PASS', 'FAIL'):
            print(f"  {k}:  {n} / {total}")

    # ---- 失败明细 (默认打印, 不需要 -v) ----
    failed = [r for r in results if r['overall'] in ('FAIL', 'ERROR')]
    if failed:
        print("\n===== Failures =====")
        for r in failed:
            print(f"\n[{r['op_name']}]")
            issues = _collect_issues(r)
            if issues:
                for it in issues:
                    print(f"  - {it}")
            elif r.get('note'):
                print(f"  - {r['note']}")
            else:
                print(f"  - (no detail captured; rerun with -v for traceback)")

    return 0 if summary.get('FAIL', 0) == 0 and summary.get('ERROR', 0) == 0 else 1


def _collect_issues(result_dict: Dict[str, Any]) -> List[str]:
    """从 result dict 里收集面向用户的失败原因列表 (跨所有失败层)"""
    out: List[str] = []
    for l in result_dict.get('layers', []):
        if l.get('status') in ('FAIL', 'WARN'):
            out.extend(l.get('issues') or [])
    return out


# ===========================================================================
# 单算子模式输出
# ===========================================================================

def render_single(result: CheckResult) -> None:
    print("=" * 72)
    print(f"  Op: {result.op_name}")
    print("=" * 72)
    for l in result.layers:
        print(f"  [{l.name}] {l.status}  {l.detail}")
    print("-" * 72)
    print(f"  RESULT: {result.overall}")
    if result.note:
        print(f"  note: {result.note}")
    print("=" * 72)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="一把跑 L0+L1+L2+L3 防 hack 检查 (默认批量模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", nargs='?',
                    help="submissions 目录 (扫目录下所有 *.txt)")
    # 单算子模式下隐藏的两个位置参数 (--single 触发时使用)
    ap.add_argument("submission", nargs='?', help=argparse.SUPPRESS)
    ap.add_argument("op_name", nargs='?', help=argparse.SUPPRESS)
    # 用户可见参数
    ap.add_argument("--ref-root", default=None,
                    help=f"reference 树根目录 (默认 {DEFAULT_REF_ROOT})")
    ap.add_argument("--op", default=None,
                    help="只检测指定算子 (op 名 = 提交文件去 .txt 后缀)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="详细输出")
    # 内部 / 调试参数 (隐藏)
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-subprocess", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # ---- 单算子模式 ----
    if args.single or (args.path and args.submission):
        if not (args.path and args.submission):
            print("usage (single): check_hack.py --single <ref.py> <sub.txt> [<op_name>]",
                  file=sys.stderr)
            return 2
        ref = args.path
        sub = args.submission
        op_name = args.op_name or Path(ref).stem
        try:
            result = check_one(ref, sub, op_name, verbose=args.verbose)
        except Exception as e:
            result = CheckResult(op_name, [], 'ERROR', note=f"{type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()
        render_single(result)
        if args.json:
            print(f"__RESULT_JSON__:{json.dumps(result.to_dict(), default=str)}")
        return 0 if result.overall == 'PASS' else 1

    # ---- 批量模式 (默认) ----
    if not args.path:
        ap.print_help()
        return 2
    p = Path(args.path)
    if not p.is_dir():
        print(f"FATAL: {args.path} is not a directory.\n"
              f"For single-op mode use:  check_hack.py --single <ref.py> <sub.txt>",
              file=sys.stderr)
        return 2
    ref_root = Path(args.ref_root) if args.ref_root else None
    return check_batch(p, ref_root=ref_root, only_op=args.op,
                       use_subprocess=not args.no_subprocess,
                       verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
