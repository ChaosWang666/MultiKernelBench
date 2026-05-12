#!/usr/bin/env python3
"""检查AscendC融合算子代码中的Hack情况.

判定依据:
1) Hack-by-fallback (Python 层回退): 文件名所标识的待融合算子中, 任意一个没有
   用AscendC在自定义kernel中实现, 而是在 model_src.forward 里直接调用了
   PyTorch (nn.X / F.x / torch.x / 张量算术等) 现成API完成.
2) Hack-by-binding (C++ 层回退): 在 python_bind_src 的 C++ 绑定函数中调用
   LibTorch 的 at::xxx / torch::xxx 算子 (如 at::linear, at::max_pool3d) 来
   完成本应在 kernel 内做的运算. 这种 Hack 不会出现在 model_src 里, 仅看 Python
   端看不出来, 必须穿透到 C++ 绑定才能识别.
3) Hack-by-missing-kernel: 即使前两条都没命中, 如果 kernel_src 里也没有对应算子
   的 AscendC 原语 (例如声称 conv2d 却没有 Matmul/Axpy, 声称 sigmoid 却没有
   Sigmoid 调用), 说明该算子根本没在 kernel 内实现.

任一种情况命中, 该 token 即不通过; 文件只要存在不通过的 token, 整体判 ⚠️ FAIL.

输出: 在指定路径生成一份 markdown 报告.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


# ---------- 1. 文件名 → 待融合算子 token 列表 ----------

# 贪心匹配的 token 集合 (长 token 排在前面, 防止被切碎)
GREEDY_TOKENS = [
    'conv_transpose2d', 'conv_transpose3d',
    'convtranspose2d', 'convtranspose3d',
    'conv_transpose', 'convtranspose',
    'conv2d', 'conv3d', 'conv1d',
    'global_avg_pool', 'globalavgpool',
    'avg_pool', 'avgpool',
    'max_pool', 'maxpool',
    'batch_norm', 'batchnorm',
    'group_norm', 'groupnorm',
    'instance_norm', 'instancenorm',
    'layer_norm', 'layernorm',
    'log_sum_exp', 'logsumexp',
    'hard_swish', 'hardswish',
    'hard_tanh', 'hardtanh',
    'leaky_relu', 'leakyrelu',
    'residual_add',
    'bias_add', 'biasadd',
    'gemm', 'bmm', 'matmul', 'linear',
    'softmax', 'sigmoid', 'tanh', 'relu', 'gelu', 'mish', 'swish',
    'dropout',
    'mean', 'sum', 'min', 'max',
    'clamp',
    'divide', 'multiply',
    'scaling', 'scale',
    'subtract',
    'add',
    'activation',
]


def parse_filename_tokens(stem: str) -> list[str]:
    """把 'conv2d_relu_bias_add' 分解为 ['conv2d', 'relu', 'bias_add']."""
    remaining = stem
    tokens: list[str] = []
    while remaining:
        matched = None
        for tok in GREEDY_TOKENS:
            if remaining == tok or remaining.startswith(tok + '_'):
                matched = tok
                break
        if matched is None:
            head, _, remaining = remaining.partition('_')
            tokens.append(head)
            continue
        tokens.append(matched)
        remaining = remaining[len(matched):].lstrip('_')
    return tokens


# ---------- 2. nn 模块类名 → 规范操作标签 ----------

MODULE_TO_OP: dict[str, str] = {
    'Conv1d': 'conv1d',
    'Conv2d': 'conv2d',
    'Conv3d': 'conv3d',
    'ConvTranspose1d': 'conv_transpose1d',
    'ConvTranspose2d': 'conv_transpose2d',
    'ConvTranspose3d': 'conv_transpose3d',
    'Linear': 'linear',
    'BatchNorm1d': 'batch_norm',
    'BatchNorm2d': 'batch_norm',
    'BatchNorm3d': 'batch_norm',
    'GroupNorm': 'group_norm',
    'InstanceNorm1d': 'instance_norm',
    'InstanceNorm2d': 'instance_norm',
    'InstanceNorm3d': 'instance_norm',
    'LayerNorm': 'layer_norm',
    'MaxPool1d': 'max_pool',
    'MaxPool2d': 'max_pool',
    'MaxPool3d': 'max_pool',
    'AvgPool1d': 'avg_pool',
    'AvgPool2d': 'avg_pool',
    'AvgPool3d': 'avg_pool',
    'AdaptiveAvgPool1d': 'global_avg_pool',
    'AdaptiveAvgPool2d': 'global_avg_pool',
    'AdaptiveAvgPool3d': 'global_avg_pool',
    'AdaptiveMaxPool1d': 'global_max_pool',
    'AdaptiveMaxPool2d': 'global_max_pool',
    'AdaptiveMaxPool3d': 'global_max_pool',
    'Dropout': 'dropout',
    'Softmax': 'softmax',
    'Sigmoid': 'sigmoid',
    'Tanh': 'tanh',
    'ReLU': 'relu',
    'LeakyReLU': 'leaky_relu',
    'GELU': 'gelu',
    'Mish': 'mish',
    'Hardswish': 'hard_swish',
    'Hardtanh': 'hardtanh',
}

# 直接函数/方法调用 → 规范操作标签
DIRECT_DETECTORS: list[tuple[str, str]] = [
    (r'\btorch\.matmul\b',                  'linear'),
    (r'\btorch\.bmm\b',                     'linear'),
    (r'\btorch\.mm\b',                      'linear'),
    (r'\bF\.linear\b',                      'linear'),
    (r'\bF\.batch_norm\b',                  'batch_norm'),
    (r'\bF\.group_norm\b',                  'group_norm'),
    (r'\bF\.instance_norm\b',               'instance_norm'),
    (r'\bF\.layer_norm\b',                  'layer_norm'),
    (r'\bF\.max_pool\d?d\b',                'max_pool'),
    (r'\bF\.avg_pool\d?d\b',                'avg_pool'),
    (r'\bF\.adaptive_avg_pool\d?d\b',       'global_avg_pool'),
    (r'\bF\.dropout\b',                     'dropout'),
    (r'\btorch\.softmax\b',                 'softmax'),
    (r'\bF\.softmax\b',                     'softmax'),
    (r'\btorch\.sigmoid\b',                 'sigmoid'),
    (r'\.sigmoid\(\)',                      'sigmoid'),
    (r'\bF\.sigmoid\b',                     'sigmoid'),
    (r'\btorch\.tanh\b',                    'tanh'),
    (r'\.tanh\(\)',                         'tanh'),
    (r'\bF\.relu\b',                        'relu'),
    (r'\btorch\.relu\b',                    'relu'),
    (r'\bF\.leaky_relu\b',                  'leaky_relu'),
    (r'\bF\.gelu\b',                        'gelu'),
    (r'torch\.nn\.functional\.gelu\b',      'gelu'),
    (r'\bF\.mish\b',                        'mish'),
    (r'\bF\.hardswish\b',                   'hard_swish'),
    (r'\bF\.hardtanh\b',                    'hardtanh'),
    (r'\btorch\.clamp\b',                   'clamp'),
    (r'\.clamp\(',                          'clamp'),
    (r'\btorch\.logsumexp\b',               'log_sum_exp'),
    (r'\btorch\.mean\b',                    'mean'),
    (r'\.mean\(',                           'mean'),
    (r'\btorch\.sum\b',                     'sum'),
    (r'\.sum\(',                            'sum'),
    (r'\btorch\.min\b',                     'min'),
    (r'\.min\(\s*dim',                      'min'),
    (r'\btorch\.max\b',                     'max'),
    (r'\.max\(\s*dim',                      'max'),
]

# token -> 哪些 op_label 算作该 token 的 hack
TOKEN_TO_OPS: dict[str, list[str]] = {
    'conv2d':           ['conv2d'],
    'conv3d':           ['conv3d'],
    'conv1d':           ['conv1d'],
    'conv_transpose2d': ['conv_transpose2d'],
    'conv_transpose3d': ['conv_transpose3d'],
    'convtranspose2d':  ['conv_transpose2d'],
    'convtranspose3d':  ['conv_transpose3d'],
    'conv_transpose':   ['conv_transpose2d', 'conv_transpose3d'],
    'convtranspose':    ['conv_transpose2d', 'conv_transpose3d'],
    'gemm':             ['linear'],
    'matmul':           ['linear'],
    'bmm':              ['linear'],
    'linear':           ['linear'],
    'batch_norm':       ['batch_norm'],
    'batchnorm':        ['batch_norm'],
    'group_norm':       ['group_norm'],
    'groupnorm':        ['group_norm'],
    'instance_norm':    ['instance_norm'],
    'instancenorm':     ['instance_norm'],
    'layer_norm':       ['layer_norm'],
    'layernorm':        ['layer_norm'],
    'max_pool':         ['max_pool'],
    'maxpool':          ['max_pool'],
    'avg_pool':         ['avg_pool'],
    'avgpool':          ['avg_pool'],
    'global_avg_pool':  ['global_avg_pool', 'mean'],
    'globalavgpool':    ['global_avg_pool', 'mean'],
    'softmax':          ['softmax'],
    'sigmoid':          ['sigmoid'],
    'tanh':             ['tanh'],
    'relu':             ['relu'],
    'gelu':             ['gelu'],
    'mish':             ['mish'],
    'swish':            ['sigmoid'],
    'hard_swish':       ['hard_swish'],
    'hardswish':        ['hard_swish'],
    'hard_tanh':        ['hardtanh'],
    'hardtanh':         ['hardtanh'],
    'leaky_relu':       ['leaky_relu'],
    'leakyrelu':        ['leaky_relu'],
    'dropout':          ['dropout'],
    'clamp':            ['clamp'],
    'log_sum_exp':      ['log_sum_exp'],
    'logsumexp':        ['log_sum_exp'],
    'mean':             ['mean', 'global_avg_pool'],
    'sum':              ['sum'],
    'min':              ['min'],
    'max':              ['max'],
    'divide':           ['divide'],
    'multiply':         ['multiply'],
    'scale':            ['multiply'],
    'scaling':          ['multiply'],
    'subtract':         ['subtract'],
    'add':              ['add'],
    'bias_add':         ['add'],
    'biasadd':          ['add'],
    'residual_add':     ['add'],
    'activation':       [],  # 太宽泛, 不作判定
}


# ---------- 3. kernel_src 内 AscendC 原语探测 ----------

# 每个 token 对应一组 AscendC 原语片段, kernel_src 出现任意一个即视为已实现.
# value 为 None 表示该 token 缺乏可靠的字符串特征, 不做 kernel 端检查.
# 原语用正则匹配, 避免 'Gelu(' 误匹配 'KernelGemmMaxSubGelu()' 这样的类名/构造函数.
# 约定: 函数原语统一写成 r'\bName\b\s*<?' 或 r'\bName\b\s*\(' 等, 通过函数生成.

def _func_call_re(name: str) -> str:
    """匹配 `name(` 或 `name<`, 但前面必须是非标识符字符 (单词边界)."""
    return r'(?<![A-Za-z_0-9])' + re.escape(name) + r'\s*[(<]'


KERNEL_PRIMITIVES: dict[str, list[str] | None] = {
    'conv2d':           [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('Axpy')],
    'conv3d':           [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('Axpy')],
    'conv1d':           [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('Axpy')],
    'conv_transpose2d': [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('Axpy')],
    'conv_transpose3d': [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('Axpy')],
    'convtranspose2d':  [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('Axpy')],
    'convtranspose3d':  [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('Axpy')],
    'conv_transpose':   [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('Axpy')],
    'convtranspose':    [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('Axpy')],
    'gemm':             [_func_call_re('Matmul'), r'\bmm\.Iterate'],
    'matmul':           [_func_call_re('Matmul'), r'\bmm\.Iterate'],
    'bmm':              [_func_call_re('Matmul'), r'\bmm\.Iterate', _func_call_re('BatchMatmul')],
    'linear':           [_func_call_re('Matmul'), r'\bmm\.Iterate'],
    'relu':             [_func_call_re('Relu'), _func_call_re('Maxs')],
    'leaky_relu':       [_func_call_re('LeakyRelu')],
    'leakyrelu':        [_func_call_re('LeakyRelu')],
    'sigmoid':          [_func_call_re('Sigmoid')],
    'tanh':             [_func_call_re('Tanh')],
    # softmax 既接受 AscendC::Softmax, 也接受 DoSoftmax/RowSoftmax 等用户自定义包装
    'softmax':          [r'(?<![A-Za-z_0-9])[A-Za-z]*[Ss]oftmax\s*\('],
    'hard_tanh':        [_func_call_re('Hardtanh'), _func_call_re('HardTanh')],
    'hardtanh':         [_func_call_re('Hardtanh'), _func_call_re('HardTanh')],
    # dropout 推理时常被实现成恒等或 Muls(mask), 缺可靠特征, 不判定
    'dropout':          None,
    'sum':              [_func_call_re('ReduceSum'), _func_call_re('BlockReduceSum'),
                         _func_call_re('WholeReduceSum'), _func_call_re('RepeatReduceSum')],
    'mean':             [_func_call_re('ReduceSum'), _func_call_re('BlockReduceSum'),
                         _func_call_re('WholeReduceSum'), _func_call_re('Mean')],
    'min':              [_func_call_re('Min'), _func_call_re('Mins'),
                         _func_call_re('ReduceMin'), _func_call_re('BlockReduceMin'),
                         _func_call_re('WholeReduceMin')],
    'max':              [_func_call_re('Max'), _func_call_re('Maxs'),
                         _func_call_re('ReduceMax'), _func_call_re('BlockReduceMax'),
                         _func_call_re('WholeReduceMax')],
    'max_pool':         [_func_call_re('ReduceMax'), _func_call_re('BlockReduceMax'),
                         _func_call_re('WholeReduceMax'), _func_call_re('Max'),
                         _func_call_re('Maxs')],
    'maxpool':          [_func_call_re('ReduceMax'), _func_call_re('BlockReduceMax'),
                         _func_call_re('WholeReduceMax'), _func_call_re('Max'),
                         _func_call_re('Maxs')],
    'avg_pool':         [_func_call_re('ReduceSum'), _func_call_re('BlockReduceSum'),
                         _func_call_re('WholeReduceSum'), _func_call_re('Sum')],
    'avgpool':          [_func_call_re('ReduceSum'), _func_call_re('BlockReduceSum'),
                         _func_call_re('WholeReduceSum'), _func_call_re('Sum')],
    'global_avg_pool':  [_func_call_re('ReduceSum'), _func_call_re('BlockReduceSum'),
                         _func_call_re('WholeReduceSum'), _func_call_re('Sum'),
                         _func_call_re('Mean')],
    'globalavgpool':    [_func_call_re('ReduceSum'), _func_call_re('BlockReduceSum'),
                         _func_call_re('WholeReduceSum'), _func_call_re('Sum'),
                         _func_call_re('Mean')],
    'divide':           [_func_call_re('Div'), _func_call_re('Divs'),
                         _func_call_re('Reciprocal')],
    'multiply':         [_func_call_re('Mul'), _func_call_re('Muls')],
    'scale':            [_func_call_re('Mul'), _func_call_re('Muls')],
    'scaling':          [_func_call_re('Mul'), _func_call_re('Muls')],
    # subtract 可能通过 Sub/Subs, 也可能通过 Adds(x, -c) (常量减法的常见写法)
    'subtract':         [_func_call_re('Sub'), _func_call_re('Subs'), _func_call_re('Adds')],
    # bias add 可能直接 Add/Adds, 也可能折叠进 Matmul 的 SetBias
    'add':              [_func_call_re('Add'), _func_call_re('Adds'), _func_call_re('SetBias')],
    'bias_add':         [_func_call_re('Add'), _func_call_re('Adds'), _func_call_re('SetBias')],
    'biasadd':          [_func_call_re('Add'), _func_call_re('Adds'), _func_call_re('SetBias')],
    'residual_add':     [_func_call_re('Add')],
    # 复合算子用 check_kernel_for_token 里的专门逻辑
    'gelu':             [],  # handled specially
    'mish':             [],
    'swish':            [],
    'hard_swish':       [],
    'hardswish':        [],
    'clamp':            [],
    'log_sum_exp':      [],
    'logsumexp':        [],
    # 太弱, 不做 kernel 端判定
    'batch_norm':       None,
    'batchnorm':        None,
    'group_norm':       None,
    'groupnorm':        None,
    'instance_norm':    None,
    'instancenorm':     None,
    'layer_norm':       None,
    'layernorm':        None,
    'activation':       None,
}


def _has_call(kernel_src: str, name: str) -> bool:
    return re.search(_func_call_re(name), kernel_src) is not None


def check_kernel_for_token(kernel_src: str, token: str) -> bool | None:
    """判断 kernel_src 是否实现了 token 对应的算子.

    返回:
      True  -- 检测到对应原语, 认为已实现
      False -- 没找到任何对应原语, 认为未实现
      None  -- 该 token 缺乏可靠 kernel 端特征, 不做判定
    """
    # 复合算子
    if token == 'gelu':
        if any(_has_call(kernel_src, n) for n in ('Gelu', 'GELU', 'Erf')):
            return True
        return _has_call(kernel_src, 'Tanh')  # tanh-近似, 弱信号
    if token == 'mish':
        if _has_call(kernel_src, 'Mish'):
            return True
        has_tanh = _has_call(kernel_src, 'Tanh')
        has_softplus = any(_has_call(kernel_src, n) for n in ('Exp', 'Softplus', 'Log1p', 'Ln'))
        return has_tanh and has_softplus
    if token == 'swish':
        if any(_has_call(kernel_src, n) for n in ('Swish', 'Silu')):
            return True
        return _has_call(kernel_src, 'Sigmoid')
    if token in ('hard_swish', 'hardswish'):
        if any(_has_call(kernel_src, n) for n in ('Hardswish', 'HardSwish')):
            return True
        return (_has_call(kernel_src, 'Adds')
                and _has_call(kernel_src, 'Mins')
                and _has_call(kernel_src, 'Maxs'))
    if token == 'clamp':
        if _has_call(kernel_src, 'Clamp'):
            return True
        return _has_call(kernel_src, 'Mins') and _has_call(kernel_src, 'Maxs')
    if token in ('log_sum_exp', 'logsumexp'):
        has_exp = _has_call(kernel_src, 'Exp')
        has_log = any(_has_call(kernel_src, n) for n in ('Ln', 'Log'))
        return has_exp and has_log

    primitives = KERNEL_PRIMITIVES.get(token)
    if primitives is None:
        return None
    if len(primitives) == 0:
        return None
    return any(re.search(p, kernel_src) for p in primitives)


# ---------- 4. python_bind_src 内 LibTorch 调用探测 ----------

# 每个 token 对应可能出现在 C++ 绑定 (at::xxx / torch::xxx) 中的 LibTorch 函数名.
# 命中 = "该 token 实际上是在 C++ 绑定层用 LibTorch 算的, 不算 AscendC 实现".
LIBTORCH_OPS: dict[str, list[str]] = {
    'conv2d':           ['conv2d'],
    'conv3d':           ['conv3d'],
    'conv1d':           ['conv1d'],
    'conv_transpose2d': ['conv_transpose2d'],
    'conv_transpose3d': ['conv_transpose3d'],
    'convtranspose2d':  ['conv_transpose2d'],
    'convtranspose3d':  ['conv_transpose3d'],
    'conv_transpose':   ['conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d'],
    'convtranspose':    ['conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d'],
    'gemm':             ['linear', 'matmul', 'mm', 'addmm'],
    'matmul':           ['linear', 'matmul', 'mm', 'addmm'],
    'bmm':              ['bmm', 'matmul'],
    'linear':           ['linear', 'addmm'],
    'batch_norm':       ['batch_norm', 'native_batch_norm'],
    'batchnorm':        ['batch_norm', 'native_batch_norm'],
    'group_norm':       ['group_norm', 'native_group_norm'],
    'groupnorm':        ['group_norm', 'native_group_norm'],
    'instance_norm':    ['instance_norm'],
    'instancenorm':     ['instance_norm'],
    'layer_norm':       ['layer_norm', 'native_layer_norm'],
    'layernorm':        ['layer_norm', 'native_layer_norm'],
    'max_pool':         ['max_pool1d', 'max_pool2d', 'max_pool3d',
                         'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d'],
    'maxpool':          ['max_pool1d', 'max_pool2d', 'max_pool3d',
                         'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d'],
    'avg_pool':         ['avg_pool1d', 'avg_pool2d', 'avg_pool3d',
                         'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d'],
    'avgpool':          ['avg_pool1d', 'avg_pool2d', 'avg_pool3d',
                         'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d'],
    'global_avg_pool':  ['adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d', 'mean'],
    'globalavgpool':    ['adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d', 'mean'],
    'softmax':          ['softmax', 'log_softmax', '_softmax'],
    'sigmoid':          ['sigmoid'],
    'tanh':             ['tanh'],
    'relu':             ['relu', 'relu_'],
    'leaky_relu':       ['leaky_relu', 'leaky_relu_'],
    'leakyrelu':        ['leaky_relu', 'leaky_relu_'],
    'gelu':             ['gelu'],
    'mish':             ['mish'],
    'swish':            ['silu'],
    'hard_swish':       ['hardswish', 'hardswish_'],
    'hardswish':        ['hardswish', 'hardswish_'],
    'hard_tanh':        ['hardtanh', 'hardtanh_'],
    'hardtanh':         ['hardtanh', 'hardtanh_'],
    'dropout':          ['dropout'],
    'clamp':            ['clamp', 'clip'],
    'log_sum_exp':      ['logsumexp'],
    'logsumexp':        ['logsumexp'],
    'mean':             ['mean'],
    'sum':              ['sum'],
    'min':              ['min', 'amin', 'minimum'],
    'max':              ['max', 'amax', 'maximum'],
    'divide':           ['div', 'divide', 'true_divide'],
    'multiply':         ['mul', 'multiply'],
    'scale':            ['mul', 'multiply'],
    'scaling':          ['mul', 'multiply'],
    'subtract':         ['sub', 'subtract'],
    'add':              ['add'],
    'bias_add':         ['add'],
    'biasadd':          ['add'],
    'residual_add':     ['add'],
    'activation':       [],
}


def detect_libtorch_calls(bind_src: str, tokens: list[str]) -> list[dict]:
    """在 python_bind_src 内查找 at::<op>(/torch::<op>( 调用, 与 token 匹配.

    返回若干 {token, op_name, line_no, code} 记录.
    """
    if not bind_src:
        return []
    hits: list[dict] = []
    used_lines: set[tuple[str, int, str]] = set()
    for tok in tokens:
        candidates = LIBTORCH_OPS.get(tok, [])
        for op in candidates:
            # 匹配 at::op( 或 torch::op(, 注意 op 名首尾要 word boundary
            pattern = r'\b(?:at|torch)::' + re.escape(op) + r'\b\s*\('
            for ln_no, line in enumerate(bind_src.split('\n'), start=1):
                if re.search(pattern, line):
                    key = (tok, ln_no, op)
                    if key in used_lines:
                        continue
                    used_lines.add(key)
                    hits.append({
                        'token': tok,
                        'op_name': f'at::{op}',
                        'line_no': ln_no,
                        'code': line.strip(),
                    })
                    break  # 每个 token 取一次证据即可
            if any(h['token'] == tok for h in hits):
                break
    return hits


# ---------- 5. 文本解析 ----------

def extract_string_block(text: str, var_name: str) -> str | None:
    """从 `var_name='''...'''` 或 `var_name=\"\"\"...\"\"\"` 中取出字符串内容."""
    for quote in ("'''", '"""'):
        pattern = re.escape(var_name) + r"\s*=\s*" + re.escape(quote) + r"(.*?)" + re.escape(quote)
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1)
    return None


def extract_method_body(model_src: str, method_name: str) -> str:
    """提取指定方法的完整缩进体."""
    lines = model_src.split('\n')
    body: list[str] = []
    in_method = False
    base_indent = -1
    for line in lines:
        if not in_method:
            m = re.match(r'(\s*)def\s+' + re.escape(method_name) + r'\s*\(', line)
            if m:
                in_method = True
                base_indent = len(m.group(1))
            continue
        if line.strip() == '':
            body.append(line)
            continue
        cur_indent = len(line) - len(line.lstrip())
        if cur_indent <= base_indent:
            break
        body.append(line)
    return '\n'.join(body)


def parse_init_attrs(model_src: str) -> dict[str, str]:
    """扫描 __init__, 返回 self.<attr> → op_label 的映射 (基于 nn.X 类型)."""
    body = extract_method_body(model_src, '__init__')
    mapping: dict[str, str] = {}
    pat = re.compile(
        r'self\.(\w+)\s*=\s*(?:torch\.)?nn\.([A-Za-z][A-Za-z0-9_]*)'
    )
    for m in pat.finditer(body):
        attr, klass = m.group(1), m.group(2)
        op = MODULE_TO_OP.get(klass)
        if op is not None:
            mapping[attr] = op
    return mapping


def strip_custom_calls(forward_body: str) -> str:
    """删除 custom_ops_lib.<x>_custom(...) 的完整调用 (含跨行参数列表).

    保留行号: 用空格填充而不是删除字符, 这样行号偏移不会发生变化.
    """
    out_chars = list(forward_body)
    for m in re.finditer(r'custom_ops_lib\.\w+\s*\(', forward_body):
        start = m.start()
        i = m.end() - 1
        depth = 0
        while i < len(forward_body):
            ch = forward_body[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        end = i if i < len(forward_body) else len(forward_body) - 1
        for j in range(start, end + 1):
            if out_chars[j] != '\n':
                out_chars[j] = ' '
    return ''.join(out_chars)


def _strip_strings_and_comments(line: str) -> str:
    """简单移除单行注释和字符串字面值, 减少 ' * ' / ' + ' 等的误判."""
    no_comment = re.sub(r'#.*$', '', line)
    no_str = re.sub(r"'[^']*'|\"[^\"]*\"", '""', no_comment)
    return no_str


def _has_arith(line: str, op: str) -> bool:
    """检测是否含算术运算 op ∈ {+, -, *, /}.

    必须左右两侧都是变量/字面值/属性访问, 排除复合赋值 (+= 等)、unary 等.
    """
    clean = _strip_strings_and_comments(line)
    if op == '+':
        pat = r'(?<![+\-*/=<>!])(?<!\+)\s+\+\s+(?!=)'
    elif op == '-':
        pat = r'(?<![+\-*/=<>!])(?<!-)\s+-\s+(?!=)'
    elif op == '*':
        pat = r'(?<![+\-*/=<>!])(?<!\*)\s+\*\s+(?!=|\*)'
    elif op == '/':
        pat = r'(?<![+\-*/=<>!])(?<!/)\s+/\s+(?!=|/)'
    else:
        return False
    return re.search(pat, clean) is not None


# ---------- 6. Hack 判定 ----------

def detect_hacks(forward_body: str, tokens: list[str], init_attrs: dict[str, str]) -> list[dict]:
    """返回若干 hack 记录, 每条形如 {token, op_label, line_no, code}."""
    stripped = strip_custom_calls(forward_body)
    detected: list[tuple[str, int, str]] = []  # (op_label, line_no, code)

    for ln_no, line in enumerate(stripped.split('\n'), start=1):
        if not line.strip():
            continue
        clean = _strip_strings_and_comments(line)

        # self.<attr>(... 形式调用 → 查 __init__ 的 nn.X 映射
        for m in re.finditer(r'self\.(\w+)\s*\(', clean):
            attr = m.group(1)
            if attr in init_attrs:
                detected.append((init_attrs[attr], ln_no, line.strip()))

        # 直接 torch.X / F.X / .x() 风格调用
        for regex, op_label in DIRECT_DETECTORS:
            if re.search(regex, clean):
                detected.append((op_label, ln_no, line.strip()))

        # 算术运算
        if _has_arith(clean, '*'):
            detected.append(('multiply', ln_no, line.strip()))
        if _has_arith(clean, '/'):
            detected.append(('divide', ln_no, line.strip()))
        if _has_arith(clean, '-'):
            detected.append(('subtract', ln_no, line.strip()))
        if _has_arith(clean, '+'):
            detected.append(('add', ln_no, line.strip()))

    hacks: list[dict] = []
    # 同一行同一 op 只算一次证据, 但同一 token 之间允许复用证据是不合理的;
    # 因此采用 (token, line_no, op_label) 作为去重键, 每个 token 仅取首条命中.
    seen_token: set[str] = set()
    for tok in tokens:
        if tok in seen_token:
            # 即使同名 token 重复 (如 mish_mish), 仍允许多次匹配不同证据行
            pass
        expected_ops = TOKEN_TO_OPS.get(tok, [])
        if not expected_ops:
            continue
        consumed_evidence: tuple[int, str] | None = None
        for expected in expected_ops:
            for op_label, ln_no, code in detected:
                if op_label != expected:
                    continue
                key = (ln_no, op_label)
                # 跳过被同一文件内之前 token 已消费的相同证据
                if any(h['token'] != tok and h['line_no'] == ln_no and h['op_label'] == op_label for h in hacks):
                    continue
                # 跳过当前 token 已经取过的同一证据
                if any(h['token'] == tok and h['line_no'] == ln_no and h['op_label'] == op_label for h in hacks):
                    continue
                hacks.append({
                    'token': tok,
                    'op_label': op_label,
                    'line_no': ln_no,
                    'code': code,
                })
                consumed_evidence = key
                break
            if consumed_evidence is not None:
                break
        seen_token.add(tok)
    return hacks


# ---------- 7. 主流程 + 报告 ----------

def check_file(path: Path) -> dict:
    text = path.read_text(encoding='utf-8', errors='replace')
    tokens = parse_filename_tokens(path.stem)
    model_src = extract_string_block(text, 'model_src') or ''
    kernel_src = extract_string_block(text, 'kernel_src') or ''
    bind_src = extract_string_block(text, 'python_bind_src') or ''
    init_attrs = parse_init_attrs(model_src) if model_src else {}
    forward = extract_method_body(model_src, 'forward') if model_src else ''

    # 检查 A: forward 内 PyTorch 回退
    hacks = detect_hacks(forward, tokens, init_attrs) if forward else []
    hacked_tokens = set(h['token'] for h in hacks)

    # 检查 C: python_bind_src 内 LibTorch at:: / torch:: 调用
    bind_hacks = detect_libtorch_calls(bind_src, tokens)
    bind_hacked_tokens = set(h['token'] for h in bind_hacks)

    # 检查 B: kernel_src 内 AscendC 原语
    kernel_status: dict[str, bool | None] = {}
    for tok in tokens:
        kernel_status[tok] = check_kernel_for_token(kernel_src, tok) if kernel_src else None

    # 已经被检查 A 或 C 命中的 token, 不再重复算 "kernel-missing"
    already_failed = hacked_tokens | bind_hacked_tokens
    missing_tokens = [
        tok for tok in tokens
        if tok not in already_failed and kernel_status.get(tok) is False
    ]

    failed_tokens = sorted(hacked_tokens | bind_hacked_tokens | set(missing_tokens))
    clean_tokens = [t for t in tokens if t not in failed_tokens]
    return {
        'file': path.name,
        'tokens': tokens,
        'init_attrs': init_attrs,
        'forward': forward,
        'kernel_src': kernel_src,
        'bind_src': bind_src,
        'hacks': hacks,
        'hacked_tokens': sorted(hacked_tokens),
        'bind_hacks': bind_hacks,
        'bind_hacked_tokens': sorted(bind_hacked_tokens),
        'kernel_status': kernel_status,
        'missing_tokens': missing_tokens,
        'failed_tokens': failed_tokens,
        'clean_tokens': clean_tokens,
    }


def _kernel_status_str(status: bool | None) -> str:
    if status is True:
        return '✓'
    if status is False:
        return '✗'
    return '—'


def render_report(results: list[dict], in_dir: Path) -> str:
    total = len(results)
    fallback_files = sum(1 for r in results if r['hacked_tokens'])
    bind_files = sum(1 for r in results if r['bind_hacked_tokens'])
    missing_files = sum(1 for r in results if r['missing_tokens'])
    failed_files = sum(1 for r in results if r['failed_tokens'])
    clean_files = total - failed_files

    lines: list[str] = []
    lines.append('# AscendC 融合算子 Hack 检测报告')
    lines.append('')
    lines.append(f'- 扫描目录: `{in_dir}`')
    lines.append(f'- 文件总数 (排除 `.raw.txt`): **{total}**')
    lines.append(f'- 存在失败 token 的文件总数: **{failed_files}**')
    lines.append(f'  - 检查 A 命中 (Python `forward` 内回退 PyTorch): **{fallback_files}**')
    lines.append(f'  - 检查 B 命中 (kernel 内找不到对应 AscendC 原语): **{missing_files}**')
    lines.append(f'  - 检查 C 命中 (C++ 绑定层调用 `at::xxx` / `torch::xxx`): **{bind_files}**')
    lines.append(f'- 完全通过的文件数: **{clean_files}**')
    lines.append('')
    lines.append('## 判定规则')
    lines.append('')
    lines.append('文件名按下划线切分为若干"待融合算子"token (例如 '
                 '`conv2d_relu_bias_add` → `[conv2d, relu, bias_add]`). 对每个 token 做三项独立检查:')
    lines.append('')
    lines.append('**检查 A — `model_src.forward` 内是否回退 PyTorch (Hack-by-fallback)**')
    lines.append('')
    lines.append('1. 从 `__init__` 抽取 `self.<attr> = nn.<Klass>(...)`, 建立 `self.<attr>` → 操作类型映射;')
    lines.append('2. 在 `forward` 中抹除 `custom_ops_lib.<x>_custom(...)` 调用 (含跨行参数), '
                 '在剩余文本中扫描 `self.<attr>(...)`、`torch.X`、`F.X`、`.x()`、二元算术 `+ - * /`;')
    lines.append('3. 任一 token 与之命中, 即视为该 token 走了 PyTorch 回退.')
    lines.append('')
    lines.append('**检查 B — `kernel_src` 内是否有对应 AscendC 原语 (Hack-by-missing-kernel)**')
    lines.append('')
    lines.append('对每个 token 在 `kernel_src` 中搜索 AscendC 原语 (粗略对应关系):')
    lines.append('')
    lines.append('- `conv* / matmul / gemm / bmm / linear` → `Matmul<...>`, `mm.Iterate`, `Axpy(`;')
    lines.append('- `relu`/`sigmoid`/`tanh`/`gelu`/`mish`/`softmax`/`leaky_relu` → 同名 `AscendC::<Op>` 原语;')
    lines.append('- `mish` 也接受 `Tanh(`+(`Exp(` 或 `Ln(`/`Softplus(`) 的手写实现;')
    lines.append('- `hard_swish` 接受 `Hardswish(` 或 `Adds(`+`Mins(`+`Maxs(` 的手写六分之一夹断;')
    lines.append('- `clamp` 接受 `Clamp(` 或 `Mins(`+`Maxs(`;')
    lines.append('- `log_sum_exp` 要求 `Exp(`+(`Ln(`/`Log(`);')
    lines.append('- `softmax` 也接受 `DoSoftmax(`/`RowSoftmax(` 等用户包装;')
    lines.append('- `subtract` 接受 `Sub/Subs/Adds`, `add/bias_add` 接受 `Add/Adds/SetBias`;')
    lines.append('- `max_pool`/`avg_pool`/`mean`/`sum` 接受对应的 `Reduce*` / `Max*` / `Min*` / `Sum(`;')
    lines.append('- 归一化类 (`batch_norm`/`group_norm`/`instance_norm`/`layer_norm`) 与 '
                 '`activation`/`dropout` 缺可靠字符串特征, 标记为 `—` (不判定).')
    lines.append('')
    lines.append('**检查 C — `python_bind_src` C++ 绑定层是否调用 LibTorch (Hack-by-binding)**')
    lines.append('')
    lines.append('在 `python_bind_src` 字符串里扫描形如 `at::<op>(` 或 `torch::<op>(` 的调用, '
                 '`<op>` 与 token 对应的 LibTorch 函数名匹配 (例如 token `matmul`/`gemm`/`linear` '
                 '→ `at::linear`、`at::matmul`、`at::mm`、`at::addmm`; token `max_pool` '
                 '→ `at::max_pool[123]d`; token `conv2d` → `at::conv2d` 等). 命中即认为该 token '
                 '实际由 LibTorch 在绑定层算出, 不算 AscendC 实现.')
    lines.append('')
    lines.append('**最终失败口径**: 任一 token 触发 A / B / C 任意一项, 即视为该 token 失败. '
                 '文件只要有一个 token 失败, 整体判 ⚠️ FAIL.')
    lines.append('')

    lines.append('## 汇总')
    lines.append('')
    lines.append('| 文件 | 待融合算子 | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 缺失 | 判定 |')
    lines.append('| --- | --- | --- | --- | --- | --- |')
    for r in sorted(results, key=lambda x: x['file']):
        verdict = '⚠️ FAIL' if r['failed_tokens'] else '✅ PASS'
        ht = ', '.join(r['hacked_tokens']) if r['hacked_tokens'] else '—'
        bt = ', '.join(r['bind_hacked_tokens']) if r['bind_hacked_tokens'] else '—'
        mt = ', '.join(r['missing_tokens']) if r['missing_tokens'] else '—'
        lines.append(f'| `{r["file"]}` | {", ".join(r["tokens"])} | {ht} | {bt} | {mt} | {verdict} |')
    lines.append('')

    lines.append('## 详细列表 (按 token 维度展开, 仅展示 FAIL 文件)')
    lines.append('')
    for r in sorted(results, key=lambda x: x['file']):
        if not r['failed_tokens']:
            continue
        lines.append(f'### {r["file"]}')
        lines.append(f'- 待融合算子: `{r["tokens"]}`')
        lines.append(f'- 通过的算子: `{r["clean_tokens"]}`')
        lines.append(f'- 失败的算子: `{r["failed_tokens"]}`')
        lines.append('')
        lines.append('| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |')
        lines.append('| --- | --- | --- | --- | --- |')
        hack_by_tok = {h['token']: h for h in r['hacks']}
        bind_by_tok = {h['token']: h for h in r['bind_hacks']}
        for tok in r['tokens']:
            fb = hack_by_tok.get(tok)
            fb_str = '✗ ' + f'`{fb["code"]}`' if fb else '—'
            bd = bind_by_tok.get(tok)
            bd_str = '✗ ' + f'`{bd["code"]}`' if bd else '—'
            k_status = r['kernel_status'].get(tok)
            k_str = _kernel_status_str(k_status)
            if tok in r['failed_tokens']:
                conclusion = '⚠️ FAIL'
            else:
                conclusion = '✅ PASS' if k_status is True else 'ℹ️ 不可判定'
            lines.append(f'| `{tok}` | {fb_str} | {bd_str} | {k_str} | {conclusion} |')
        lines.append('')

    lines.append('## 完全通过的文件')
    lines.append('')
    pass_files = [r['file'] for r in sorted(results, key=lambda x: x['file']) if not r['failed_tokens']]
    if pass_files:
        for name in pass_files:
            lines.append(f'- `{name}`')
    else:
        lines.append('（无）')
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='AscendC fused operator hack detector')
    parser.add_argument('--input-dir', default='.', help='待扫描的目录 (默认: 当前目录)')
    parser.add_argument('--output', default='hack_report.md', help='输出 markdown 报告路径')
    args = parser.parse_args()

    in_dir = Path(args.input_dir).resolve()
    txt_files = sorted(
        p for p in in_dir.glob('*.txt')
        if not p.name.endswith('.raw.txt')
    )
    if not txt_files:
        print(f'未在 {in_dir} 中找到 .txt 文件')
        return

    results = [check_file(p) for p in txt_files]

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = in_dir / out_path
    out_path.write_text(render_report(results, in_dir), encoding='utf-8')

    total = len(results)
    fallback = sum(1 for r in results if r['hacked_tokens'])
    bind = sum(1 for r in results if r['bind_hacked_tokens'])
    missing = sum(1 for r in results if r['missing_tokens'])
    failed = sum(1 for r in results if r['failed_tokens'])
    print(f'报告已写入 {out_path}')
    print(f'文件总数: {total}, 失败文件: {failed} '
          f'(A forward 回退 {fallback}, C 绑定层 {bind}, B kernel 缺失 {missing}), '
          f'完全通过: {total - failed}')


if __name__ == '__main__':
    main()
