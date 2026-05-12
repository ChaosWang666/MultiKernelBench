# AscendC 融合算子 Hack 检测报告

- 扫描目录: `/data2/z00810025/MultiKernelBench/output/ascendc/add_shot/0.0-1.0/claude-code-static/run0`
- 文件总数 (排除 `.raw.txt`): **97**
- 存在失败 token 的文件总数: **92**
  - 检查 A 命中 (Python `forward` 内回退 PyTorch): **80**
  - 检查 B 命中 (kernel 内找不到对应 AscendC 原语): **49**
  - 检查 C 命中 (C++ 绑定层调用 `at::xxx` / `torch::xxx`): **5**
- 完全通过的文件数: **5**

## 判定规则

文件名按下划线切分为若干"待融合算子"token (例如 `conv2d_relu_bias_add` → `[conv2d, relu, bias_add]`). 对每个 token 做三项独立检查:

**检查 A — `model_src.forward` 内是否回退 PyTorch (Hack-by-fallback)**

1. 从 `__init__` 抽取 `self.<attr> = nn.<Klass>(...)`, 建立 `self.<attr>` → 操作类型映射;
2. 在 `forward` 中抹除 `custom_ops_lib.<x>_custom(...)` 调用 (含跨行参数), 在剩余文本中扫描 `self.<attr>(...)`、`torch.X`、`F.X`、`.x()`、二元算术 `+ - * /`;
3. 任一 token 与之命中, 即视为该 token 走了 PyTorch 回退.

**检查 B — `kernel_src` 内是否有对应 AscendC 原语 (Hack-by-missing-kernel)**

对每个 token 在 `kernel_src` 中搜索 AscendC 原语 (粗略对应关系):

- `conv* / matmul / gemm / bmm / linear` → `Matmul<...>`, `mm.Iterate`, `Axpy(`;
- `relu`/`sigmoid`/`tanh`/`gelu`/`mish`/`softmax`/`leaky_relu` → 同名 `AscendC::<Op>` 原语;
- `mish` 也接受 `Tanh(`+(`Exp(` 或 `Ln(`/`Softplus(`) 的手写实现;
- `hard_swish` 接受 `Hardswish(` 或 `Adds(`+`Mins(`+`Maxs(` 的手写六分之一夹断;
- `clamp` 接受 `Clamp(` 或 `Mins(`+`Maxs(`;
- `log_sum_exp` 要求 `Exp(`+(`Ln(`/`Log(`);
- `softmax` 也接受 `DoSoftmax(`/`RowSoftmax(` 等用户包装;
- `subtract` 接受 `Sub/Subs/Adds`, `add/bias_add` 接受 `Add/Adds/SetBias`;
- `max_pool`/`avg_pool`/`mean`/`sum` 接受对应的 `Reduce*` / `Max*` / `Min*` / `Sum(`;
- 归一化类 (`batch_norm`/`group_norm`/`instance_norm`/`layer_norm`) 与 `activation`/`dropout` 缺可靠字符串特征, 标记为 `—` (不判定).

**检查 C — `python_bind_src` C++ 绑定层是否调用 LibTorch (Hack-by-binding)**

在 `python_bind_src` 字符串里扫描形如 `at::<op>(` 或 `torch::<op>(` 的调用, `<op>` 与 token 对应的 LibTorch 函数名匹配 (例如 token `matmul`/`gemm`/`linear` → `at::linear`、`at::matmul`、`at::mm`、`at::addmm`; token `max_pool` → `at::max_pool[123]d`; token `conv2d` → `at::conv2d` 等). 命中即认为该 token 实际由 LibTorch 在绑定层算出, 不算 AscendC 实现.

**最终失败口径**: 任一 token 触发 A / B / C 任意一项, 即视为该 token 失败. 文件只要有一个 token 失败, 整体判 ⚠️ FAIL.

## 汇总

| 文件 | 待融合算子 | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 缺失 | 判定 |
| --- | --- | --- | --- | --- | --- |
| `bmm_instance_norm_sum_residual_add_multiply.txt` | bmm, instance_norm, sum, residual_add, multiply | bmm | — | — | ⚠️ FAIL |
| `conv2d_activation_batch_norm.txt` | conv2d, activation, batch_norm | batch_norm, conv2d | — | — | ⚠️ FAIL |
| `conv2d_add_scale_sigmoid_group_norm.txt` | conv2d, add, scale, sigmoid, group_norm | — | conv2d | sigmoid | ⚠️ FAIL |
| `conv2d_avg_pool_sigmoid_sum.txt` | conv2d, avg_pool, sigmoid, sum | avg_pool, conv2d | — | sigmoid | ⚠️ FAIL |
| `conv2d_batch_norm_scaling.txt` | conv2d, batch_norm, scaling | batch_norm, conv2d | — | — | ⚠️ FAIL |
| `conv2d_divide_leaky_relu.txt` | conv2d, divide, leaky_relu | conv2d | — | divide, leaky_relu | ⚠️ FAIL |
| `conv2d_group_norm_scale_max_pool_clamp.txt` | conv2d, group_norm, scale, max_pool, clamp | clamp, conv2d, group_norm, max_pool, scale | — | — | ⚠️ FAIL |
| `conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp.txt` | conv2d, group_norm, tanh, hard_swish, residual_add, log_sum_exp | conv2d, group_norm, log_sum_exp | — | — | ⚠️ FAIL |
| `conv2d_hard_swish_relu.txt` | conv2d, hard_swish, relu | — | — | — | ✅ PASS |
| `conv2d_instance_norm_divide.txt` | conv2d, instance_norm, divide | conv2d | — | divide | ⚠️ FAIL |
| `conv2d_min_add_multiply.txt` | conv2d, min, add, multiply | conv2d | — | — | ⚠️ FAIL |
| `conv2d_min_tanh_tanh.txt` | conv2d, min, tanh, tanh | conv2d, min | — | — | ⚠️ FAIL |
| `conv2d_mish_mish.txt` | conv2d, mish, mish | conv2d | — | mish, mish | ⚠️ FAIL |
| `conv2d_multiply_leaky_relu_gelu.txt` | conv2d, multiply, leaky_relu, gelu | conv2d, multiply | — | leaky_relu | ⚠️ FAIL |
| `conv2d_relu_bias_add.txt` | conv2d, relu, bias_add | conv2d | — | — | ⚠️ FAIL |
| `conv2d_relu_hard_swish.txt` | conv2d, relu, hard_swish | — | conv2d | — | ⚠️ FAIL |
| `conv2d_scaling_min.txt` | conv2d, scaling, min | conv2d, scaling | — | — | ⚠️ FAIL |
| `conv2d_subtract_hard_swish_max_pool_mish.txt` | conv2d, subtract, hard_swish, max_pool, mish | conv2d, max_pool | — | mish | ⚠️ FAIL |
| `conv2d_subtract_subtract_mish.txt` | conv2d, subtract, subtract, mish | conv2d, subtract | — | — | ⚠️ FAIL |
| `conv2d_subtract_tanh_subtract_avg_pool.txt` | conv2d, subtract, tanh, subtract, avg_pool | avg_pool, conv2d | — | — | ⚠️ FAIL |
| `conv2d_tanh_scaling_bias_add_max.txt` | conv2d, tanh, scaling, bias_add, max | bias_add, conv2d, scaling | — | tanh, max | ⚠️ FAIL |
| `conv3d_divide_max_global_avg_pool_bias_add_sum.txt` | conv3d, divide, max, global_avg_pool, bias_add, sum | conv3d, divide, global_avg_pool | — | max | ⚠️ FAIL |
| `conv3d_group_norm_mean.txt` | conv3d, group_norm, mean | conv3d, group_norm | — | — | ⚠️ FAIL |
| `conv3d_group_norm_min_clamp_dropout.txt` | conv3d, group_norm, min, clamp, dropout | — | — | conv3d, min, clamp | ⚠️ FAIL |
| `conv3d_hardswish_relu_softmax_mean.txt` | conv3d, hardswish, relu, softmax, mean | conv3d, mean | — | softmax | ⚠️ FAIL |
| `conv3d_leaky_relu_sum_clamp_gelu.txt` | conv3d, leaky_relu, sum, clamp, gelu | — | — | leaky_relu, sum | ⚠️ FAIL |
| `conv3d_max_log_sum_exp_relu.txt` | conv3d, max, log_sum_exp, relu | conv3d | — | — | ⚠️ FAIL |
| `conv3d_min_softmax.txt` | conv3d, min, softmax | conv3d, min, softmax | — | — | ⚠️ FAIL |
| `conv3d_mish_tanh.txt` | conv3d, mish, tanh | conv3d | — | — | ⚠️ FAIL |
| `conv3d_multiply_instance_norm_clamp_multiply_max.txt` | conv3d, multiply, instance_norm, clamp, multiply, max | conv3d, max | — | — | ⚠️ FAIL |
| `conv3d_relu_leaky_relu_gelu_sigmoid_bias_add.txt` | conv3d, relu, leaky_relu, gelu, sigmoid, bias_add | conv3d | — | leaky_relu, sigmoid | ⚠️ FAIL |
| `conv3d_scaling_tanh_multiply_sigmoid.txt` | conv3d, scaling, tanh, multiply, sigmoid | conv3d | — | tanh, sigmoid | ⚠️ FAIL |
| `conv3d_softmax_max_pool_max_pool.txt` | conv3d, softmax, max_pool, max_pool | conv3d, softmax | — | max_pool, max_pool | ⚠️ FAIL |
| `conv_transpose2d_add_min_gelu_multiply.txt` | conv_transpose2d, add, min, gelu, multiply | conv_transpose2d | — | — | ⚠️ FAIL |
| `conv_transpose2d_bias_add_clamp_scaling_clamp_divide.txt` | conv_transpose2d, bias_add, clamp, scaling, clamp, divide | conv_transpose2d | — | scaling, divide | ⚠️ FAIL |
| `conv_transpose2d_gelu_group_norm.txt` | conv_transpose2d, gelu, group_norm | conv_transpose2d, group_norm | — | gelu | ⚠️ FAIL |
| `conv_transpose2d_max_pool_hardtanh_mean_tanh.txt` | conv_transpose2d, max_pool, hardtanh, mean, tanh | conv_transpose2d, max_pool | — | hardtanh, tanh | ⚠️ FAIL |
| `conv_transpose2d_min_sum_gelu_add.txt` | conv_transpose2d, min, sum, gelu, add | conv_transpose2d | — | sum, gelu | ⚠️ FAIL |
| `conv_transpose2d_mish_add_hardtanh_scaling.txt` | conv_transpose2d, mish, add, hardtanh, scaling | conv_transpose2d | — | hardtanh | ⚠️ FAIL |
| `conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean.txt` | conv_transpose2d, multiply, global_avg_pool, global_avg_pool, mean | conv_transpose2d, multiply | — | — | ⚠️ FAIL |
| `conv_transpose2d_subtract_tanh.txt` | conv_transpose2d, subtract, tanh | conv_transpose2d | — | — | ⚠️ FAIL |
| `conv_transpose3d_add_hard_swish.txt` | conv_transpose3d, add, hard_swish | conv_transpose3d | — | — | ⚠️ FAIL |
| `conv_transpose3d_avg_pool_clamp_softmax_multiply.txt` | conv_transpose3d, avg_pool, clamp, softmax, multiply | avg_pool, clamp, conv_transpose3d, multiply, softmax | — | — | ⚠️ FAIL |
| `conv_transpose3d_batch_norm_avg_pool_avg_pool.txt` | conv_transpose3d, batch_norm, avg_pool, avg_pool | avg_pool, batch_norm, conv_transpose3d | — | — | ⚠️ FAIL |
| `conv_transpose3d_batch_norm_subtract.txt` | conv_transpose3d, batch_norm, subtract | batch_norm, conv_transpose3d | — | — | ⚠️ FAIL |
| `conv_transpose3d_clamp_min_divide.txt` | conv_transpose3d, clamp, min, divide | conv_transpose3d | — | clamp, min, divide | ⚠️ FAIL |
| `conv_transpose3d_leaky_relu_multiply_leaky_relu_max.txt` | conv_transpose3d, leaky_relu, multiply, leaky_relu, max | — | conv_transpose3d | max | ⚠️ FAIL |
| `conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max.txt` | conv_transpose3d, log_sum_exp, hard_swish, subtract, clamp, max | conv_transpose3d | — | hard_swish, clamp | ⚠️ FAIL |
| `conv_transpose3d_max_max_sum.txt` | conv_transpose3d, max, max, sum | conv_transpose3d | — | max, max, sum | ⚠️ FAIL |
| `conv_transpose3d_max_pool_softmax_subtract_swish_max.txt` | conv_transpose3d, max_pool, softmax, subtract, swish, max | conv_transpose3d, max_pool | — | softmax, swish | ⚠️ FAIL |
| `conv_transpose3d_multiply_max_global_avg_pool_clamp.txt` | conv_transpose3d, multiply, max, global_avg_pool, clamp | conv_transpose3d, global_avg_pool, multiply | — | — | ⚠️ FAIL |
| `conv_transpose3d_scale_batch_norm_global_avg_pool.txt` | conv_transpose3d, scale, batch_norm, global_avg_pool | batch_norm, conv_transpose3d, scale | — | — | ⚠️ FAIL |
| `conv_transpose3d_scaling_avg_pool_bias_add_scaling.txt` | conv_transpose3d, scaling, avg_pool, bias_add, scaling | avg_pool, conv_transpose3d, scaling | — | — | ⚠️ FAIL |
| `conv_transpose3d_softmax_sigmoid.txt` | conv_transpose3d, softmax, sigmoid | conv_transpose3d | — | softmax, sigmoid | ⚠️ FAIL |
| `conv_transpose3d_sum_layer_norm_avg_pool_gelu.txt` | conv_transpose3d, sum, layer_norm, avg_pool, gelu | avg_pool, conv_transpose3d, layer_norm | — | sum | ⚠️ FAIL |
| `conv_transpose3d_sum_residual_add_multiply_residual_add.txt` | conv_transpose3d, sum, residual_add, multiply, residual_add | conv_transpose3d | — | sum | ⚠️ FAIL |
| `conv_transpose3d_swish_group_norm_hard_swish.txt` | conv_transpose3d, swish, group_norm, hard_swish | conv_transpose3d, group_norm, swish | — | — | ⚠️ FAIL |
| `convtranspose2d_batchnorm_tanh_maxpool_groupnorm.txt` | convtranspose2d, batchnorm, tanh, maxpool, groupnorm | batchnorm, convtranspose2d, groupnorm, maxpool | — | tanh | ⚠️ FAIL |
| `convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply.txt` | convtranspose2d, globalavgpool, biasadd, logsumexp, sum, multiply | convtranspose2d | — | multiply | ⚠️ FAIL |
| `convtranspose2d_softmax_biasadd_scaling_sigmoid.txt` | convtranspose2d, softmax, biasadd, scaling, sigmoid | biasadd, convtranspose2d, softmax | — | — | ⚠️ FAIL |
| `convtranspose3d_mean_add_softmax_tanh_scaling.txt` | convtranspose3d, mean, add, softmax, tanh, scaling | add, convtranspose3d, mean, scaling | — | softmax | ⚠️ FAIL |
| `convtranspose3d_relu_groupnorm.txt` | convtranspose3d, relu, groupnorm | convtranspose3d, groupnorm | — | — | ⚠️ FAIL |
| `gemm_add_relu.txt` | gemm, add, relu | — | — | — | ✅ PASS |
| `gemm_batch_norm_gelu_group_norm_mean_relu.txt` | gemm, batch_norm, gelu, group_norm, mean, relu | batch_norm, gemm | — | mean | ⚠️ FAIL |
| `gemm_batch_norm_scaling_softmax.txt` | gemm, batch_norm, scaling, softmax | batch_norm, gemm, scaling | — | — | ⚠️ FAIL |
| `gemm_bias_add_hardtanh_mish_group_norm.txt` | gemm, bias_add, hardtanh, mish, group_norm | — | — | gemm, hardtanh, mish | ⚠️ FAIL |
| `gemm_divide_sum_scaling.txt` | gemm, divide, sum, scaling | divide, scaling, sum | — | gemm | ⚠️ FAIL |
| `gemm_group_norm_hardtanh.txt` | gemm, group_norm, hardtanh | gemm | — | hardtanh | ⚠️ FAIL |
| `gemm_group_norm_min_bias_add.txt` | gemm, group_norm, min, bias_add | gemm, group_norm | — | — | ⚠️ FAIL |
| `gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu.txt` | gemm, log_sum_exp, leaky_relu, leaky_relu, gelu, gelu | gemm | — | leaky_relu, leaky_relu | ⚠️ FAIL |
| `gemm_max_subtract_gelu.txt` | gemm, max, subtract, gelu | — | — | gemm, max, subtract, gelu | ⚠️ FAIL |
| `gemm_multiply_leakyrelu.txt` | gemm, multiply, leakyrelu | — | — | — | ✅ PASS |
| `gemm_relu_divide.txt` | gemm, relu, divide | — | — | divide | ⚠️ FAIL |
| `gemm_scale_batch_norm.txt` | gemm, scale, batch_norm | batch_norm, gemm | — | — | ⚠️ FAIL |
| `gemm_scale_batchnorm.txt` | gemm, scale, batchnorm | — | — | gemm | ⚠️ FAIL |
| `gemm_scaling_hard_tanh_gelu.txt` | gemm, scaling, hard_tanh, gelu | — | — | hard_tanh | ⚠️ FAIL |
| `gemm_sigmoid_scaling_residual_add.txt` | gemm, sigmoid, scaling, residual_add | — | — | — | ✅ PASS |
| `gemm_sigmoid_sum_log_sum_exp.txt` | gemm, sigmoid, sum, log_sum_exp | gemm, sigmoid | — | log_sum_exp | ⚠️ FAIL |
| `gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add.txt` | gemm, subtract, global_avg_pool, log_sum_exp, gelu, residual_add | gemm | — | log_sum_exp, gelu, residual_add | ⚠️ FAIL |
| `gemm_swish_divide_clamp_tanh_clamp.txt` | gemm, swish, divide, clamp, tanh, clamp | gemm | — | divide | ⚠️ FAIL |
| `matmul_add_swish_tanh_gelu_hardtanh.txt` | matmul, add, swish, tanh, gelu, hardtanh | matmul | — | hardtanh | ⚠️ FAIL |
| `matmul_avg_pool_gelu_scale_max.txt` | matmul, avg_pool, gelu, scale, max | matmul | — | avg_pool | ⚠️ FAIL |
| `matmul_batch_norm_bias_add_divide_swish.txt` | matmul, batch_norm, bias_add, divide, swish | batch_norm, bias_add, divide, matmul | — | swish | ⚠️ FAIL |
| `matmul_divide_gelu.txt` | matmul, divide, gelu | divide | — | — | ⚠️ FAIL |
| `matmul_dropout_mean_softmax.txt` | matmul, dropout, mean, softmax | — | — | — | ✅ PASS |
| `matmul_gelu_softmax.txt` | matmul, gelu, softmax | gelu, matmul | — | — | ⚠️ FAIL |
| `matmul_group_norm_leaky_relu_sum.txt` | matmul, group_norm, leaky_relu, sum | matmul | — | — | ⚠️ FAIL |
| `matmul_max_pool_sum_scale.txt` | matmul, max_pool, sum, scale | matmul, max_pool, scale | — | — | ⚠️ FAIL |
| `matmul_min_subtract.txt` | matmul, min, subtract | matmul | — | — | ⚠️ FAIL |
| `matmul_mish_mish.txt` | matmul, mish, mish | matmul | — | — | ⚠️ FAIL |
| `matmul_scale_residual_add_clamp_log_sum_exp_mish.txt` | matmul, scale, residual_add, clamp, log_sum_exp, mish | matmul | — | residual_add, mish | ⚠️ FAIL |
| `matmul_scaling_residual_add.txt` | matmul, scaling, residual_add | residual_add, scaling | — | — | ⚠️ FAIL |
| `matmul_sigmoid_sum.txt` | matmul, sigmoid, sum | — | matmul | — | ⚠️ FAIL |
| `matmul_subtract_multiply_relu.txt` | matmul, subtract, multiply, relu | — | matmul | — | ⚠️ FAIL |
| `matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp.txt` | matmul, sum, max, avg_pool, log_sum_exp, log_sum_exp | sum | — | matmul, max, log_sum_exp, log_sum_exp | ⚠️ FAIL |
| `matmul_swish_scaling.txt` | matmul, swish, scaling | matmul | — | — | ⚠️ FAIL |
| `matmul_swish_sum_group_norm.txt` | matmul, swish, sum, group_norm | matmul | — | swish | ⚠️ FAIL |

## 详细列表 (按 token 维度展开, 仅展示 FAIL 文件)

### bmm_instance_norm_sum_residual_add_multiply.txt
- 待融合算子: `['bmm', 'instance_norm', 'sum', 'residual_add', 'multiply']`
- 通过的算子: `['instance_norm', 'sum', 'residual_add', 'multiply']`
- 失败的算子: `['bmm']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `bmm` | ✗ `x = self.bmm(x)` | — | ✗ | ⚠️ FAIL |
| `instance_norm` | — | — | — | ℹ️ 不可判定 |
| `sum` | — | — | ✓ | ✅ PASS |
| `residual_add` | — | — | ✓ | ✅ PASS |
| `multiply` | — | — | ✓ | ✅ PASS |

### conv2d_activation_batch_norm.txt
- 待融合算子: `['conv2d', 'activation', 'batch_norm']`
- 通过的算子: `['activation']`
- 失败的算子: `['batch_norm', 'conv2d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `activation` | — | — | — | ℹ️ 不可判定 |
| `batch_norm` | ✗ `x = self.bn(x)` | — | — | ⚠️ FAIL |

### conv2d_add_scale_sigmoid_group_norm.txt
- 待融合算子: `['conv2d', 'add', 'scale', 'sigmoid', 'group_norm']`
- 通过的算子: `['add', 'scale', 'group_norm']`
- 失败的算子: `['conv2d', 'sigmoid']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | — | ✗ `at::Tensor t = at::conv2d(x, conv_w, conv_b).contiguous();` | ✗ | ⚠️ FAIL |
| `add` | — | — | ✓ | ✅ PASS |
| `scale` | — | — | ✓ | ✅ PASS |
| `sigmoid` | — | — | ✗ | ⚠️ FAIL |
| `group_norm` | — | — | — | ℹ️ 不可判定 |

### conv2d_avg_pool_sigmoid_sum.txt
- 待融合算子: `['conv2d', 'avg_pool', 'sigmoid', 'sum']`
- 通过的算子: `['sum']`
- 失败的算子: `['avg_pool', 'conv2d', 'sigmoid']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `avg_pool` | ✗ `x = self.avg_pool(x)` | — | ✓ | ⚠️ FAIL |
| `sigmoid` | — | — | ✗ | ⚠️ FAIL |
| `sum` | — | — | ✓ | ✅ PASS |

### conv2d_batch_norm_scaling.txt
- 待融合算子: `['conv2d', 'batch_norm', 'scaling']`
- 通过的算子: `['scaling']`
- 失败的算子: `['batch_norm', 'conv2d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `batch_norm` | ✗ `x = self.bn(x)` | — | — | ⚠️ FAIL |
| `scaling` | — | — | ✓ | ✅ PASS |

### conv2d_divide_leaky_relu.txt
- 待融合算子: `['conv2d', 'divide', 'leaky_relu']`
- 通过的算子: `[]`
- 失败的算子: `['conv2d', 'divide', 'leaky_relu']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `divide` | — | — | ✗ | ⚠️ FAIL |
| `leaky_relu` | — | — | ✗ | ⚠️ FAIL |

### conv2d_group_norm_scale_max_pool_clamp.txt
- 待融合算子: `['conv2d', 'group_norm', 'scale', 'max_pool', 'clamp']`
- 通过的算子: `[]`
- 失败的算子: `['clamp', 'conv2d', 'group_norm', 'max_pool', 'scale']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `group_norm` | ✗ `x = self.group_norm(x)` | — | — | ⚠️ FAIL |
| `scale` | ✗ `x = x * self.scale` | — | ✗ | ⚠️ FAIL |
| `max_pool` | ✗ `x = self.maxpool(x)` | — | ✓ | ⚠️ FAIL |
| `clamp` | ✗ `x = torch.clamp(x, self.clamp_min, self.clamp_max)` | — | ✓ | ⚠️ FAIL |

### conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp.txt
- 待融合算子: `['conv2d', 'group_norm', 'tanh', 'hard_swish', 'residual_add', 'log_sum_exp']`
- 通过的算子: `['tanh', 'hard_swish', 'residual_add']`
- 失败的算子: `['conv2d', 'group_norm', 'log_sum_exp']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x_conv = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `group_norm` | ✗ `x_norm = self.group_norm(x_conv)` | — | — | ⚠️ FAIL |
| `tanh` | — | — | ✓ | ✅ PASS |
| `hard_swish` | — | — | ✓ | ✅ PASS |
| `residual_add` | — | — | ✓ | ✅ PASS |
| `log_sum_exp` | ✗ `return torch.logsumexp(fused, dim=1, keepdim=True)` | — | ✗ | ⚠️ FAIL |

### conv2d_instance_norm_divide.txt
- 待融合算子: `['conv2d', 'instance_norm', 'divide']`
- 通过的算子: `['instance_norm']`
- 失败的算子: `['conv2d', 'divide']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `instance_norm` | — | — | — | ℹ️ 不可判定 |
| `divide` | — | — | ✗ | ⚠️ FAIL |

### conv2d_min_add_multiply.txt
- 待融合算子: `['conv2d', 'min', 'add', 'multiply']`
- 通过的算子: `['min', 'add', 'multiply']`
- 失败的算子: `['conv2d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `min` | — | — | ✓ | ✅ PASS |
| `add` | — | — | ✓ | ✅ PASS |
| `multiply` | — | — | ✓ | ✅ PASS |

### conv2d_min_tanh_tanh.txt
- 待融合算子: `['conv2d', 'min', 'tanh', 'tanh']`
- 通过的算子: `['tanh', 'tanh']`
- 失败的算子: `['conv2d', 'min']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `min` | ✗ `x = torch.min(x, dim=1, keepdim=True)[0]` | — | ✗ | ⚠️ FAIL |
| `tanh` | — | — | ✓ | ✅ PASS |
| `tanh` | — | — | ✓ | ✅ PASS |

### conv2d_mish_mish.txt
- 待融合算子: `['conv2d', 'mish', 'mish']`
- 通过的算子: `[]`
- 失败的算子: `['conv2d', 'mish']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x).contiguous()` | — | ✗ | ⚠️ FAIL |
| `mish` | — | — | ✗ | ⚠️ FAIL |
| `mish` | — | — | ✗ | ⚠️ FAIL |

### conv2d_multiply_leaky_relu_gelu.txt
- 待融合算子: `['conv2d', 'multiply', 'leaky_relu', 'gelu']`
- 通过的算子: `['gelu']`
- 失败的算子: `['conv2d', 'leaky_relu', 'multiply']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `multiply` | ✗ `x = x * self.multiplier` | — | ✓ | ⚠️ FAIL |
| `leaky_relu` | — | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✓ | ✅ PASS |

### conv2d_relu_bias_add.txt
- 待融合算子: `['conv2d', 'relu', 'bias_add']`
- 通过的算子: `['relu', 'bias_add']`
- 失败的算子: `['conv2d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `relu` | — | — | ✓ | ✅ PASS |
| `bias_add` | — | — | ✓ | ✅ PASS |

### conv2d_relu_hard_swish.txt
- 待融合算子: `['conv2d', 'relu', 'hard_swish']`
- 通过的算子: `['relu', 'hard_swish']`
- 失败的算子: `['conv2d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | — | ✗ `at::Tensor conv_out = at::conv2d(x, weight, bias).contiguous();` | ✗ | ⚠️ FAIL |
| `relu` | — | — | ✓ | ✅ PASS |
| `hard_swish` | — | — | ✓ | ✅ PASS |

### conv2d_scaling_min.txt
- 待融合算子: `['conv2d', 'scaling', 'min']`
- 通过的算子: `['min']`
- 失败的算子: `['conv2d', 'scaling']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `scaling` | ✗ `x = x * self.scale_factor` | — | ✗ | ⚠️ FAIL |
| `min` | — | — | ✓ | ✅ PASS |

### conv2d_subtract_hard_swish_max_pool_mish.txt
- 待融合算子: `['conv2d', 'subtract', 'hard_swish', 'max_pool', 'mish']`
- 通过的算子: `['subtract', 'hard_swish']`
- 失败的算子: `['conv2d', 'max_pool', 'mish']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `subtract` | — | — | ✓ | ✅ PASS |
| `hard_swish` | — | — | ✓ | ✅ PASS |
| `max_pool` | ✗ `x = self.pool(x)` | — | ✓ | ⚠️ FAIL |
| `mish` | — | — | ✗ | ⚠️ FAIL |

### conv2d_subtract_subtract_mish.txt
- 待融合算子: `['conv2d', 'subtract', 'subtract', 'mish']`
- 通过的算子: `['mish']`
- 失败的算子: `['conv2d', 'subtract']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `subtract` | ✗ `x = x - self.subtract_total` | — | ✓ | ⚠️ FAIL |
| `subtract` | ✗ `x = x - self.subtract_total` | — | ✓ | ⚠️ FAIL |
| `mish` | — | — | ✓ | ✅ PASS |

### conv2d_subtract_tanh_subtract_avg_pool.txt
- 待融合算子: `['conv2d', 'subtract', 'tanh', 'subtract', 'avg_pool']`
- 通过的算子: `['subtract', 'tanh', 'subtract']`
- 失败的算子: `['avg_pool', 'conv2d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `subtract` | — | — | ✓ | ✅ PASS |
| `tanh` | — | — | ✓ | ✅ PASS |
| `subtract` | — | — | ✓ | ✅ PASS |
| `avg_pool` | ✗ `x = self.avgpool(x)` | — | ✗ | ⚠️ FAIL |

### conv2d_tanh_scaling_bias_add_max.txt
- 待融合算子: `['conv2d', 'tanh', 'scaling', 'bias_add', 'max']`
- 通过的算子: `[]`
- 失败的算子: `['bias_add', 'conv2d', 'max', 'scaling', 'tanh']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv2d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `tanh` | — | — | ✗ | ⚠️ FAIL |
| `scaling` | ✗ `x = x * self.scaling_factor` | — | ✓ | ⚠️ FAIL |
| `bias_add` | ✗ `x = x + self.bias` | — | ✓ | ⚠️ FAIL |
| `max` | — | — | ✗ | ⚠️ FAIL |

### conv3d_divide_max_global_avg_pool_bias_add_sum.txt
- 待融合算子: `['conv3d', 'divide', 'max', 'global_avg_pool', 'bias_add', 'sum']`
- 通过的算子: `['bias_add', 'sum']`
- 失败的算子: `['conv3d', 'divide', 'global_avg_pool', 'max']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `divide` | ✗ `x = x / self.divisor` | — | ✗ | ⚠️ FAIL |
| `max` | — | — | ✗ | ⚠️ FAIL |
| `global_avg_pool` | ✗ `x = self.global_avg_pool(x)` | — | ✓ | ⚠️ FAIL |
| `bias_add` | — | — | ✓ | ✅ PASS |
| `sum` | — | — | ✓ | ✅ PASS |

### conv3d_group_norm_mean.txt
- 待融合算子: `['conv3d', 'group_norm', 'mean']`
- 通过的算子: `['mean']`
- 失败的算子: `['conv3d', 'group_norm']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `group_norm` | ✗ `x = self.group_norm(x)` | — | — | ⚠️ FAIL |
| `mean` | — | — | ✓ | ✅ PASS |

### conv3d_group_norm_min_clamp_dropout.txt
- 待融合算子: `['conv3d', 'group_norm', 'min', 'clamp', 'dropout']`
- 通过的算子: `['group_norm', 'dropout']`
- 失败的算子: `['clamp', 'conv3d', 'min']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | — | — | ✗ | ⚠️ FAIL |
| `group_norm` | — | — | — | ℹ️ 不可判定 |
| `min` | — | — | ✗ | ⚠️ FAIL |
| `clamp` | — | — | ✗ | ⚠️ FAIL |
| `dropout` | — | — | — | ℹ️ 不可判定 |

### conv3d_hardswish_relu_softmax_mean.txt
- 待融合算子: `['conv3d', 'hardswish', 'relu', 'softmax', 'mean']`
- 通过的算子: `['hardswish', 'relu']`
- 失败的算子: `['conv3d', 'mean', 'softmax']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `hardswish` | — | — | ✓ | ✅ PASS |
| `relu` | — | — | ✓ | ✅ PASS |
| `softmax` | — | — | ✗ | ⚠️ FAIL |
| `mean` | ✗ `x = torch.mean(x, dim=[2, 3, 4])` | — | ✗ | ⚠️ FAIL |

### conv3d_leaky_relu_sum_clamp_gelu.txt
- 待融合算子: `['conv3d', 'leaky_relu', 'sum', 'clamp', 'gelu']`
- 通过的算子: `['conv3d', 'clamp', 'gelu']`
- 失败的算子: `['leaky_relu', 'sum']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | — | — | ✓ | ✅ PASS |
| `leaky_relu` | — | — | ✗ | ⚠️ FAIL |
| `sum` | — | — | ✗ | ⚠️ FAIL |
| `clamp` | — | — | ✓ | ✅ PASS |
| `gelu` | — | — | ✓ | ✅ PASS |

### conv3d_max_log_sum_exp_relu.txt
- 待融合算子: `['conv3d', 'max', 'log_sum_exp', 'relu']`
- 通过的算子: `['max', 'log_sum_exp', 'relu']`
- 失败的算子: `['conv3d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `max` | — | — | ✓ | ✅ PASS |
| `log_sum_exp` | — | — | ✓ | ✅ PASS |
| `relu` | — | — | ✓ | ✅ PASS |

### conv3d_min_softmax.txt
- 待融合算子: `['conv3d', 'min', 'softmax']`
- 通过的算子: `[]`
- 失败的算子: `['conv3d', 'min', 'softmax']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `min` | ✗ `x = torch.min(x, dim=self.dim)[0]` | — | ✓ | ⚠️ FAIL |
| `softmax` | ✗ `return torch.softmax(x, dim=1)` | — | ✗ | ⚠️ FAIL |

### conv3d_mish_tanh.txt
- 待融合算子: `['conv3d', 'mish', 'tanh']`
- 通过的算子: `['mish', 'tanh']`
- 失败的算子: `['conv3d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `mish` | — | — | ✓ | ✅ PASS |
| `tanh` | — | — | ✓ | ✅ PASS |

### conv3d_multiply_instance_norm_clamp_multiply_max.txt
- 待融合算子: `['conv3d', 'multiply', 'instance_norm', 'clamp', 'multiply', 'max']`
- 通过的算子: `['multiply', 'instance_norm', 'clamp', 'multiply']`
- 失败的算子: `['conv3d', 'max']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x).contiguous()` | — | ✗ | ⚠️ FAIL |
| `multiply` | — | — | ✓ | ✅ PASS |
| `instance_norm` | — | — | — | ℹ️ 不可判定 |
| `clamp` | — | — | ✓ | ✅ PASS |
| `multiply` | — | — | ✓ | ✅ PASS |
| `max` | ✗ `x = torch.max(x, dim=1)[0]` | — | ✓ | ⚠️ FAIL |

### conv3d_relu_leaky_relu_gelu_sigmoid_bias_add.txt
- 待融合算子: `['conv3d', 'relu', 'leaky_relu', 'gelu', 'sigmoid', 'bias_add']`
- 通过的算子: `['relu', 'gelu', 'bias_add']`
- 失败的算子: `['conv3d', 'leaky_relu', 'sigmoid']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `relu` | — | — | ✓ | ✅ PASS |
| `leaky_relu` | — | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✓ | ✅ PASS |
| `sigmoid` | — | — | ✗ | ⚠️ FAIL |
| `bias_add` | — | — | ✓ | ✅ PASS |

### conv3d_scaling_tanh_multiply_sigmoid.txt
- 待融合算子: `['conv3d', 'scaling', 'tanh', 'multiply', 'sigmoid']`
- 通过的算子: `['scaling', 'multiply']`
- 失败的算子: `['conv3d', 'sigmoid', 'tanh']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x).contiguous()` | — | ✗ | ⚠️ FAIL |
| `scaling` | — | — | ✓ | ✅ PASS |
| `tanh` | — | — | ✗ | ⚠️ FAIL |
| `multiply` | — | — | ✓ | ✅ PASS |
| `sigmoid` | — | — | ✗ | ⚠️ FAIL |

### conv3d_softmax_max_pool_max_pool.txt
- 待融合算子: `['conv3d', 'softmax', 'max_pool', 'max_pool']`
- 通过的算子: `[]`
- 失败的算子: `['conv3d', 'max_pool', 'softmax']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv3d` | ✗ `x = self.conv(x)` | — | ✗ | ⚠️ FAIL |
| `softmax` | ✗ `x = torch.softmax(x, dim=1)` | — | ✗ | ⚠️ FAIL |
| `max_pool` | — | — | ✗ | ⚠️ FAIL |
| `max_pool` | — | — | ✗ | ⚠️ FAIL |

### conv_transpose2d_add_min_gelu_multiply.txt
- 待融合算子: `['conv_transpose2d', 'add', 'min', 'gelu', 'multiply']`
- 通过的算子: `['add', 'min', 'gelu', 'multiply']`
- 失败的算子: `['conv_transpose2d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `add` | — | — | ✓ | ✅ PASS |
| `min` | — | — | ✓ | ✅ PASS |
| `gelu` | — | — | ✓ | ✅ PASS |
| `multiply` | — | — | ✓ | ✅ PASS |

### conv_transpose2d_bias_add_clamp_scaling_clamp_divide.txt
- 待融合算子: `['conv_transpose2d', 'bias_add', 'clamp', 'scaling', 'clamp', 'divide']`
- 通过的算子: `['bias_add', 'clamp', 'clamp']`
- 失败的算子: `['conv_transpose2d', 'divide', 'scaling']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `bias_add` | — | — | ✓ | ✅ PASS |
| `clamp` | — | — | ✓ | ✅ PASS |
| `scaling` | — | — | ✗ | ⚠️ FAIL |
| `clamp` | — | — | ✓ | ✅ PASS |
| `divide` | — | — | ✗ | ⚠️ FAIL |

### conv_transpose2d_gelu_group_norm.txt
- 待融合算子: `['conv_transpose2d', 'gelu', 'group_norm']`
- 通过的算子: `[]`
- 失败的算子: `['conv_transpose2d', 'gelu', 'group_norm']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✗ | ⚠️ FAIL |
| `group_norm` | ✗ `x = self.group_norm(x)` | — | — | ⚠️ FAIL |

### conv_transpose2d_max_pool_hardtanh_mean_tanh.txt
- 待融合算子: `['conv_transpose2d', 'max_pool', 'hardtanh', 'mean', 'tanh']`
- 通过的算子: `['mean']`
- 失败的算子: `['conv_transpose2d', 'hardtanh', 'max_pool', 'tanh']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `max_pool` | ✗ `x = self.maxpool(x)` | — | ✓ | ⚠️ FAIL |
| `hardtanh` | — | — | ✗ | ⚠️ FAIL |
| `mean` | — | — | ✓ | ✅ PASS |
| `tanh` | — | — | ✗ | ⚠️ FAIL |

### conv_transpose2d_min_sum_gelu_add.txt
- 待融合算子: `['conv_transpose2d', 'min', 'sum', 'gelu', 'add']`
- 通过的算子: `['min', 'add']`
- 失败的算子: `['conv_transpose2d', 'gelu', 'sum']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `min` | — | — | ✓ | ✅ PASS |
| `sum` | — | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✗ | ⚠️ FAIL |
| `add` | — | — | ✓ | ✅ PASS |

### conv_transpose2d_mish_add_hardtanh_scaling.txt
- 待融合算子: `['conv_transpose2d', 'mish', 'add', 'hardtanh', 'scaling']`
- 通过的算子: `['mish', 'add', 'scaling']`
- 失败的算子: `['conv_transpose2d', 'hardtanh']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `mish` | — | — | ✓ | ✅ PASS |
| `add` | — | — | ✓ | ✅ PASS |
| `hardtanh` | — | — | ✗ | ⚠️ FAIL |
| `scaling` | — | — | ✓ | ✅ PASS |

### conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean.txt
- 待融合算子: `['conv_transpose2d', 'multiply', 'global_avg_pool', 'global_avg_pool', 'mean']`
- 通过的算子: `['global_avg_pool', 'global_avg_pool', 'mean']`
- 失败的算子: `['conv_transpose2d', 'multiply']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `multiply` | ✗ `x = x * self.multiplier` | — | ✗ | ⚠️ FAIL |
| `global_avg_pool` | — | — | ✓ | ✅ PASS |
| `global_avg_pool` | — | — | ✓ | ✅ PASS |
| `mean` | — | — | ✓ | ✅ PASS |

### conv_transpose2d_subtract_tanh.txt
- 待融合算子: `['conv_transpose2d', 'subtract', 'tanh']`
- 通过的算子: `['subtract', 'tanh']`
- 失败的算子: `['conv_transpose2d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `subtract` | — | — | ✓ | ✅ PASS |
| `tanh` | — | — | ✓ | ✅ PASS |

### conv_transpose3d_add_hard_swish.txt
- 待融合算子: `['conv_transpose3d', 'add', 'hard_swish']`
- 通过的算子: `['add', 'hard_swish']`
- 失败的算子: `['conv_transpose3d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `add` | — | — | ✓ | ✅ PASS |
| `hard_swish` | — | — | ✓ | ✅ PASS |

### conv_transpose3d_avg_pool_clamp_softmax_multiply.txt
- 待融合算子: `['conv_transpose3d', 'avg_pool', 'clamp', 'softmax', 'multiply']`
- 通过的算子: `[]`
- 失败的算子: `['avg_pool', 'clamp', 'conv_transpose3d', 'multiply', 'softmax']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `avg_pool` | ✗ `x = self.avg_pool(x)` | — | ✗ | ⚠️ FAIL |
| `clamp` | ✗ `x = torch.clamp(x, self.clamp_min, self.clamp_max)` | — | ✗ | ⚠️ FAIL |
| `softmax` | ✗ `x = torch.softmax(x, dim=2)` | — | ✗ | ⚠️ FAIL |
| `multiply` | ✗ `x_flat = x.contiguous().view(b * c, -1)` | — | ✓ | ⚠️ FAIL |

### conv_transpose3d_batch_norm_avg_pool_avg_pool.txt
- 待融合算子: `['conv_transpose3d', 'batch_norm', 'avg_pool', 'avg_pool']`
- 通过的算子: `[]`
- 失败的算子: `['avg_pool', 'batch_norm', 'conv_transpose3d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `batch_norm` | ✗ `x = self.batch_norm(x)` | — | — | ⚠️ FAIL |
| `avg_pool` | ✗ `x = self.avg_pool2(x)` | — | ✗ | ⚠️ FAIL |
| `avg_pool` | ✗ `x = self.avg_pool2(x)` | — | ✗ | ⚠️ FAIL |

### conv_transpose3d_batch_norm_subtract.txt
- 待融合算子: `['conv_transpose3d', 'batch_norm', 'subtract']`
- 通过的算子: `['subtract']`
- 失败的算子: `['batch_norm', 'conv_transpose3d']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `batch_norm` | ✗ `x = self.batch_norm(x)` | — | — | ⚠️ FAIL |
| `subtract` | — | — | ✓ | ✅ PASS |

### conv_transpose3d_clamp_min_divide.txt
- 待融合算子: `['conv_transpose3d', 'clamp', 'min', 'divide']`
- 通过的算子: `[]`
- 失败的算子: `['clamp', 'conv_transpose3d', 'divide', 'min']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `clamp` | — | — | ✗ | ⚠️ FAIL |
| `min` | — | — | ✗ | ⚠️ FAIL |
| `divide` | — | — | ✗ | ⚠️ FAIL |

### conv_transpose3d_leaky_relu_multiply_leaky_relu_max.txt
- 待融合算子: `['conv_transpose3d', 'leaky_relu', 'multiply', 'leaky_relu', 'max']`
- 通过的算子: `['leaky_relu', 'multiply', 'leaky_relu']`
- 失败的算子: `['conv_transpose3d', 'max']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | — | ✗ `auto conv_out = at::conv_transpose3d(` | ✗ | ⚠️ FAIL |
| `leaky_relu` | — | — | ✓ | ✅ PASS |
| `multiply` | — | — | ✓ | ✅ PASS |
| `leaky_relu` | — | — | ✓ | ✅ PASS |
| `max` | — | — | ✗ | ⚠️ FAIL |

### conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max.txt
- 待融合算子: `['conv_transpose3d', 'log_sum_exp', 'hard_swish', 'subtract', 'clamp', 'max']`
- 通过的算子: `['log_sum_exp', 'subtract', 'max']`
- 失败的算子: `['clamp', 'conv_transpose3d', 'hard_swish']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `log_sum_exp` | — | — | ✓ | ✅ PASS |
| `hard_swish` | — | — | ✗ | ⚠️ FAIL |
| `subtract` | — | — | ✓ | ✅ PASS |
| `clamp` | — | — | ✗ | ⚠️ FAIL |
| `max` | — | — | ✓ | ✅ PASS |

### conv_transpose3d_max_max_sum.txt
- 待融合算子: `['conv_transpose3d', 'max', 'max', 'sum']`
- 通过的算子: `[]`
- 失败的算子: `['conv_transpose3d', 'max', 'sum']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `max` | — | — | ✗ | ⚠️ FAIL |
| `max` | — | — | ✗ | ⚠️ FAIL |
| `sum` | — | — | ✗ | ⚠️ FAIL |

### conv_transpose3d_max_pool_softmax_subtract_swish_max.txt
- 待融合算子: `['conv_transpose3d', 'max_pool', 'softmax', 'subtract', 'swish', 'max']`
- 通过的算子: `['subtract', 'max']`
- 失败的算子: `['conv_transpose3d', 'max_pool', 'softmax', 'swish']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `max_pool` | ✗ `x = self.max_pool(x)` | — | ✓ | ⚠️ FAIL |
| `softmax` | — | — | ✗ | ⚠️ FAIL |
| `subtract` | — | — | ✓ | ✅ PASS |
| `swish` | — | — | ✗ | ⚠️ FAIL |
| `max` | — | — | ✓ | ✅ PASS |

### conv_transpose3d_multiply_max_global_avg_pool_clamp.txt
- 待融合算子: `['conv_transpose3d', 'multiply', 'max', 'global_avg_pool', 'clamp']`
- 通过的算子: `['max', 'clamp']`
- 失败的算子: `['conv_transpose3d', 'global_avg_pool', 'multiply']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `multiply` | ✗ `x = x * self.scale` | — | ✗ | ⚠️ FAIL |
| `max` | — | — | ✓ | ✅ PASS |
| `global_avg_pool` | ✗ `x = self.global_avg_pool(x)` | — | ✗ | ⚠️ FAIL |
| `clamp` | — | — | ✓ | ✅ PASS |

### conv_transpose3d_scale_batch_norm_global_avg_pool.txt
- 待融合算子: `['conv_transpose3d', 'scale', 'batch_norm', 'global_avg_pool']`
- 通过的算子: `['global_avg_pool']`
- 失败的算子: `['batch_norm', 'conv_transpose3d', 'scale']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `scale` | ✗ `x = x * self.scale_factor` | — | ✗ | ⚠️ FAIL |
| `batch_norm` | ✗ `x = self.batch_norm(x)` | — | — | ⚠️ FAIL |
| `global_avg_pool` | — | — | ✓ | ✅ PASS |

### conv_transpose3d_scaling_avg_pool_bias_add_scaling.txt
- 待融合算子: `['conv_transpose3d', 'scaling', 'avg_pool', 'bias_add', 'scaling']`
- 通过的算子: `['bias_add']`
- 失败的算子: `['avg_pool', 'conv_transpose3d', 'scaling']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `scaling` | ✗ `x = x * self.scale1` | — | ✓ | ⚠️ FAIL |
| `avg_pool` | ✗ `x = self.avg_pool(x)` | — | ✗ | ⚠️ FAIL |
| `bias_add` | — | — | ✓ | ✅ PASS |
| `scaling` | ✗ `x = x * self.scale1` | — | ✓ | ⚠️ FAIL |

### conv_transpose3d_softmax_sigmoid.txt
- 待融合算子: `['conv_transpose3d', 'softmax', 'sigmoid']`
- 通过的算子: `[]`
- 失败的算子: `['conv_transpose3d', 'sigmoid', 'softmax']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `softmax` | — | — | ✗ | ⚠️ FAIL |
| `sigmoid` | — | — | ✗ | ⚠️ FAIL |

### conv_transpose3d_sum_layer_norm_avg_pool_gelu.txt
- 待融合算子: `['conv_transpose3d', 'sum', 'layer_norm', 'avg_pool', 'gelu']`
- 通过的算子: `['gelu']`
- 失败的算子: `['avg_pool', 'conv_transpose3d', 'layer_norm', 'sum']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `sum` | — | — | ✗ | ⚠️ FAIL |
| `layer_norm` | ✗ `x = self.norm(x)` | — | — | ⚠️ FAIL |
| `avg_pool` | ✗ `x = self.avg_pool(x)` | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✓ | ✅ PASS |

### conv_transpose3d_sum_residual_add_multiply_residual_add.txt
- 待融合算子: `['conv_transpose3d', 'sum', 'residual_add', 'multiply', 'residual_add']`
- 通过的算子: `['residual_add', 'multiply', 'residual_add']`
- 失败的算子: `['conv_transpose3d', 'sum']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `sum` | — | — | ✗ | ⚠️ FAIL |
| `residual_add` | — | — | ✓ | ✅ PASS |
| `multiply` | — | — | ✓ | ✅ PASS |
| `residual_add` | — | — | ✓ | ✅ PASS |

### conv_transpose3d_swish_group_norm_hard_swish.txt
- 待融合算子: `['conv_transpose3d', 'swish', 'group_norm', 'hard_swish']`
- 通过的算子: `['hard_swish']`
- 失败的算子: `['conv_transpose3d', 'group_norm', 'swish']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `conv_transpose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `swish` | ✗ `x = torch.sigmoid(x) * x` | — | ✗ | ⚠️ FAIL |
| `group_norm` | ✗ `x = self.group_norm(x)` | — | — | ⚠️ FAIL |
| `hard_swish` | — | — | ✓ | ✅ PASS |

### convtranspose2d_batchnorm_tanh_maxpool_groupnorm.txt
- 待融合算子: `['convtranspose2d', 'batchnorm', 'tanh', 'maxpool', 'groupnorm']`
- 通过的算子: `[]`
- 失败的算子: `['batchnorm', 'convtranspose2d', 'groupnorm', 'maxpool', 'tanh']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `convtranspose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `batchnorm` | ✗ `x = self.batch_norm(x)` | — | — | ⚠️ FAIL |
| `tanh` | — | — | ✗ | ⚠️ FAIL |
| `maxpool` | ✗ `x = self.max_pool(x)` | — | ✗ | ⚠️ FAIL |
| `groupnorm` | ✗ `x = self.group_norm(x)` | — | — | ⚠️ FAIL |

### convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply.txt
- 待融合算子: `['convtranspose2d', 'globalavgpool', 'biasadd', 'logsumexp', 'sum', 'multiply']`
- 通过的算子: `['globalavgpool', 'biasadd', 'logsumexp', 'sum']`
- 失败的算子: `['convtranspose2d', 'multiply']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `convtranspose2d` | ✗ `y = self.conv_transpose(x).contiguous()` | — | ✗ | ⚠️ FAIL |
| `globalavgpool` | — | — | ✓ | ✅ PASS |
| `biasadd` | — | — | ✓ | ✅ PASS |
| `logsumexp` | — | — | ✓ | ✅ PASS |
| `sum` | — | — | ✓ | ✅ PASS |
| `multiply` | — | — | ✗ | ⚠️ FAIL |

### convtranspose2d_softmax_biasadd_scaling_sigmoid.txt
- 待融合算子: `['convtranspose2d', 'softmax', 'biasadd', 'scaling', 'sigmoid']`
- 通过的算子: `['scaling', 'sigmoid']`
- 失败的算子: `['biasadd', 'convtranspose2d', 'softmax']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `convtranspose2d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `softmax` | ✗ `x = torch.softmax(x, dim=1)` | — | ✗ | ⚠️ FAIL |
| `biasadd` | ✗ `x = x + self.bias` | — | ✗ | ⚠️ FAIL |
| `scaling` | — | — | ✓ | ✅ PASS |
| `sigmoid` | — | — | ✓ | ✅ PASS |

### convtranspose3d_mean_add_softmax_tanh_scaling.txt
- 待融合算子: `['convtranspose3d', 'mean', 'add', 'softmax', 'tanh', 'scaling']`
- 通过的算子: `['tanh']`
- 失败的算子: `['add', 'convtranspose3d', 'mean', 'scaling', 'softmax']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `convtranspose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `mean` | ✗ `x = x.mean(dim=2, keepdim=True)` | — | ✗ | ⚠️ FAIL |
| `add` | ✗ `x = x + self.bias` | — | ✓ | ⚠️ FAIL |
| `softmax` | — | — | ✗ | ⚠️ FAIL |
| `tanh` | — | — | ✓ | ✅ PASS |
| `scaling` | ✗ `x = x * self.scaling_factor` | — | ✓ | ⚠️ FAIL |

### convtranspose3d_relu_groupnorm.txt
- 待融合算子: `['convtranspose3d', 'relu', 'groupnorm']`
- 通过的算子: `['relu']`
- 失败的算子: `['convtranspose3d', 'groupnorm']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `convtranspose3d` | ✗ `x = self.conv_transpose(x)` | — | ✗ | ⚠️ FAIL |
| `relu` | — | — | ✓ | ✅ PASS |
| `groupnorm` | ✗ `x = self.group_norm(x)` | — | — | ⚠️ FAIL |

### gemm_batch_norm_gelu_group_norm_mean_relu.txt
- 待融合算子: `['gemm', 'batch_norm', 'gelu', 'group_norm', 'mean', 'relu']`
- 通过的算子: `['gelu', 'group_norm', 'relu']`
- 失败的算子: `['batch_norm', 'gemm', 'mean']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | ✗ `x = self.gemm(x)` | — | ✗ | ⚠️ FAIL |
| `batch_norm` | ✗ `x = self.batch_norm(x)` | — | — | ⚠️ FAIL |
| `gelu` | — | — | ✓ | ✅ PASS |
| `group_norm` | — | — | — | ℹ️ 不可判定 |
| `mean` | — | — | ✗ | ⚠️ FAIL |
| `relu` | — | — | ✓ | ✅ PASS |

### gemm_batch_norm_scaling_softmax.txt
- 待融合算子: `['gemm', 'batch_norm', 'scaling', 'softmax']`
- 通过的算子: `['softmax']`
- 失败的算子: `['batch_norm', 'gemm', 'scaling']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | ✗ `x = self.gemm(x)` | — | ✗ | ⚠️ FAIL |
| `batch_norm` | ✗ `x = self.bn(x)` | — | — | ⚠️ FAIL |
| `scaling` | ✗ `x = self.scale * x` | — | ✓ | ⚠️ FAIL |
| `softmax` | — | — | ✓ | ✅ PASS |

### gemm_bias_add_hardtanh_mish_group_norm.txt
- 待融合算子: `['gemm', 'bias_add', 'hardtanh', 'mish', 'group_norm']`
- 通过的算子: `['bias_add', 'group_norm']`
- 失败的算子: `['gemm', 'hardtanh', 'mish']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | — | — | ✗ | ⚠️ FAIL |
| `bias_add` | — | — | ✓ | ✅ PASS |
| `hardtanh` | — | — | ✗ | ⚠️ FAIL |
| `mish` | — | — | ✗ | ⚠️ FAIL |
| `group_norm` | — | — | — | ℹ️ 不可判定 |

### gemm_divide_sum_scaling.txt
- 待融合算子: `['gemm', 'divide', 'sum', 'scaling']`
- 通过的算子: `[]`
- 失败的算子: `['divide', 'gemm', 'scaling', 'sum']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | — | — | ✗ | ⚠️ FAIL |
| `divide` | ✗ `effective_w = (self.weight.sum(dim=0) * (self.scaling_factor / 2.0)).contiguous()` | — | ✗ | ⚠️ FAIL |
| `sum` | ✗ `effective_w = (self.weight.sum(dim=0) * (self.scaling_factor / 2.0)).contiguous()` | — | ✓ | ⚠️ FAIL |
| `scaling` | ✗ `effective_w = (self.weight.sum(dim=0) * (self.scaling_factor / 2.0)).contiguous()` | — | ✓ | ⚠️ FAIL |

### gemm_group_norm_hardtanh.txt
- 待融合算子: `['gemm', 'group_norm', 'hardtanh']`
- 通过的算子: `['group_norm']`
- 失败的算子: `['gemm', 'hardtanh']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | ✗ `x = self.gemm(x)` | — | ✗ | ⚠️ FAIL |
| `group_norm` | — | — | — | ℹ️ 不可判定 |
| `hardtanh` | — | — | ✗ | ⚠️ FAIL |

### gemm_group_norm_min_bias_add.txt
- 待融合算子: `['gemm', 'group_norm', 'min', 'bias_add']`
- 通过的算子: `['min', 'bias_add']`
- 失败的算子: `['gemm', 'group_norm']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | ✗ `x = self.gemm(x)` | — | ✗ | ⚠️ FAIL |
| `group_norm` | ✗ `x = self.group_norm(x)` | — | — | ⚠️ FAIL |
| `min` | — | — | ✓ | ✅ PASS |
| `bias_add` | — | — | ✓ | ✅ PASS |

### gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu.txt
- 待融合算子: `['gemm', 'log_sum_exp', 'leaky_relu', 'leaky_relu', 'gelu', 'gelu']`
- 通过的算子: `['log_sum_exp', 'gelu', 'gelu']`
- 失败的算子: `['gemm', 'leaky_relu']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | ✗ `x = self.linear(x)` | — | ✗ | ⚠️ FAIL |
| `log_sum_exp` | — | — | ✓ | ✅ PASS |
| `leaky_relu` | — | — | ✗ | ⚠️ FAIL |
| `leaky_relu` | — | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✓ | ✅ PASS |
| `gelu` | — | — | ✓ | ✅ PASS |

### gemm_max_subtract_gelu.txt
- 待融合算子: `['gemm', 'max', 'subtract', 'gelu']`
- 通过的算子: `[]`
- 失败的算子: `['gelu', 'gemm', 'max', 'subtract']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | — | — | ✗ | ⚠️ FAIL |
| `max` | — | — | ✗ | ⚠️ FAIL |
| `subtract` | — | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✗ | ⚠️ FAIL |

### gemm_relu_divide.txt
- 待融合算子: `['gemm', 'relu', 'divide']`
- 通过的算子: `['gemm', 'relu']`
- 失败的算子: `['divide']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | — | — | ✓ | ✅ PASS |
| `relu` | — | — | ✓ | ✅ PASS |
| `divide` | — | — | ✗ | ⚠️ FAIL |

### gemm_scale_batch_norm.txt
- 待融合算子: `['gemm', 'scale', 'batch_norm']`
- 通过的算子: `['scale']`
- 失败的算子: `['batch_norm', 'gemm']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | ✗ `x = self.gemm(x)` | — | ✗ | ⚠️ FAIL |
| `scale` | — | — | ✓ | ✅ PASS |
| `batch_norm` | ✗ `x = self.bn(x)` | — | — | ⚠️ FAIL |

### gemm_scale_batchnorm.txt
- 待融合算子: `['gemm', 'scale', 'batchnorm']`
- 通过的算子: `['scale', 'batchnorm']`
- 失败的算子: `['gemm']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | — | — | ✗ | ⚠️ FAIL |
| `scale` | — | — | ✓ | ✅ PASS |
| `batchnorm` | — | — | — | ℹ️ 不可判定 |

### gemm_scaling_hard_tanh_gelu.txt
- 待融合算子: `['gemm', 'scaling', 'hard_tanh', 'gelu']`
- 通过的算子: `['gemm', 'scaling', 'gelu']`
- 失败的算子: `['hard_tanh']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | — | — | ✓ | ✅ PASS |
| `scaling` | — | — | ✓ | ✅ PASS |
| `hard_tanh` | — | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✓ | ✅ PASS |

### gemm_sigmoid_sum_log_sum_exp.txt
- 待融合算子: `['gemm', 'sigmoid', 'sum', 'log_sum_exp']`
- 通过的算子: `['sum']`
- 失败的算子: `['gemm', 'log_sum_exp', 'sigmoid']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | ✗ `x = self.linear1(x)` | — | ✗ | ⚠️ FAIL |
| `sigmoid` | ✗ `x = torch.sigmoid(x)` | — | ✗ | ⚠️ FAIL |
| `sum` | — | — | ✓ | ✅ PASS |
| `log_sum_exp` | — | — | ✗ | ⚠️ FAIL |

### gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add.txt
- 待融合算子: `['gemm', 'subtract', 'global_avg_pool', 'log_sum_exp', 'gelu', 'residual_add']`
- 通过的算子: `['subtract', 'global_avg_pool']`
- 失败的算子: `['gelu', 'gemm', 'log_sum_exp', 'residual_add']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | ✗ `gemm_out = self.gemm(x)` | — | ✗ | ⚠️ FAIL |
| `subtract` | — | — | ✓ | ✅ PASS |
| `global_avg_pool` | — | — | ✓ | ✅ PASS |
| `log_sum_exp` | — | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✗ | ⚠️ FAIL |
| `residual_add` | — | — | ✗ | ⚠️ FAIL |

### gemm_swish_divide_clamp_tanh_clamp.txt
- 待融合算子: `['gemm', 'swish', 'divide', 'clamp', 'tanh', 'clamp']`
- 通过的算子: `['swish', 'clamp', 'tanh', 'clamp']`
- 失败的算子: `['divide', 'gemm']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `gemm` | ✗ `x = self.gemm(x)` | — | ✗ | ⚠️ FAIL |
| `swish` | — | — | ✓ | ✅ PASS |
| `divide` | — | — | ✗ | ⚠️ FAIL |
| `clamp` | — | — | ✓ | ✅ PASS |
| `tanh` | — | — | ✓ | ✅ PASS |
| `clamp` | — | — | ✓ | ✅ PASS |

### matmul_add_swish_tanh_gelu_hardtanh.txt
- 待融合算子: `['matmul', 'add', 'swish', 'tanh', 'gelu', 'hardtanh']`
- 通过的算子: `['add', 'swish', 'tanh', 'gelu']`
- 失败的算子: `['hardtanh', 'matmul']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.matmul(x)` | — | ✗ | ⚠️ FAIL |
| `add` | — | — | ✓ | ✅ PASS |
| `swish` | — | — | ✓ | ✅ PASS |
| `tanh` | — | — | ✓ | ✅ PASS |
| `gelu` | — | — | ✓ | ✅ PASS |
| `hardtanh` | — | — | ✗ | ⚠️ FAIL |

### matmul_avg_pool_gelu_scale_max.txt
- 待融合算子: `['matmul', 'avg_pool', 'gelu', 'scale', 'max']`
- 通过的算子: `['gelu', 'scale', 'max']`
- 失败的算子: `['avg_pool', 'matmul']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.matmul(x)` | — | ✗ | ⚠️ FAIL |
| `avg_pool` | — | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✓ | ✅ PASS |
| `scale` | — | — | ✓ | ✅ PASS |
| `max` | — | — | ✓ | ✅ PASS |

### matmul_batch_norm_bias_add_divide_swish.txt
- 待融合算子: `['matmul', 'batch_norm', 'bias_add', 'divide', 'swish']`
- 通过的算子: `[]`
- 失败的算子: `['batch_norm', 'bias_add', 'divide', 'matmul', 'swish']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.matmul(x)` | — | ✗ | ⚠️ FAIL |
| `batch_norm` | ✗ `x = self.bn(x)` | — | — | ⚠️ FAIL |
| `bias_add` | ✗ `x = x + self.bias` | — | ✓ | ⚠️ FAIL |
| `divide` | ✗ `x = x / self.divide_value` | — | ✓ | ⚠️ FAIL |
| `swish` | — | — | ✗ | ⚠️ FAIL |

### matmul_divide_gelu.txt
- 待融合算子: `['matmul', 'divide', 'gelu']`
- 通过的算子: `['matmul', 'gelu']`
- 失败的算子: `['divide']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | — | — | ✓ | ✅ PASS |
| `divide` | ✗ `self._weight_t = (self.linear.weight / self.divisor).t().contiguous()` | — | ✗ | ⚠️ FAIL |
| `gelu` | — | — | ✓ | ✅ PASS |

### matmul_gelu_softmax.txt
- 待融合算子: `['matmul', 'gelu', 'softmax']`
- 通过的算子: `['softmax']`
- 失败的算子: `['gelu', 'matmul']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.linear(x)` | — | ✗ | ⚠️ FAIL |
| `gelu` | ✗ `x = torch.nn.functional.gelu(x)` | — | ✗ | ⚠️ FAIL |
| `softmax` | — | — | ✓ | ✅ PASS |

### matmul_group_norm_leaky_relu_sum.txt
- 待融合算子: `['matmul', 'group_norm', 'leaky_relu', 'sum']`
- 通过的算子: `['group_norm', 'leaky_relu', 'sum']`
- 失败的算子: `['matmul']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.fc(x)` | — | ✗ | ⚠️ FAIL |
| `group_norm` | — | — | — | ℹ️ 不可判定 |
| `leaky_relu` | — | — | ✓ | ✅ PASS |
| `sum` | — | — | ✓ | ✅ PASS |

### matmul_max_pool_sum_scale.txt
- 待融合算子: `['matmul', 'max_pool', 'sum', 'scale']`
- 通过的算子: `['sum']`
- 失败的算子: `['matmul', 'max_pool', 'scale']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.matmul(x)` | — | ✗ | ⚠️ FAIL |
| `max_pool` | ✗ `x = self.max_pool(x.unsqueeze(1)).squeeze(1).contiguous()` | — | ✗ | ⚠️ FAIL |
| `sum` | — | — | ✓ | ✅ PASS |
| `scale` | ✗ `x = x * self.scale_factor` | — | ✗ | ⚠️ FAIL |

### matmul_min_subtract.txt
- 待融合算子: `['matmul', 'min', 'subtract']`
- 通过的算子: `['min', 'subtract']`
- 失败的算子: `['matmul']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.linear(x)` | — | ✗ | ⚠️ FAIL |
| `min` | — | — | ✓ | ✅ PASS |
| `subtract` | — | — | ✓ | ✅ PASS |

### matmul_mish_mish.txt
- 待融合算子: `['matmul', 'mish', 'mish']`
- 通过的算子: `['mish', 'mish']`
- 失败的算子: `['matmul']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.linear(x)` | — | ✗ | ⚠️ FAIL |
| `mish` | — | — | ✓ | ✅ PASS |
| `mish` | — | — | ✓ | ✅ PASS |

### matmul_scale_residual_add_clamp_log_sum_exp_mish.txt
- 待融合算子: `['matmul', 'scale', 'residual_add', 'clamp', 'log_sum_exp', 'mish']`
- 通过的算子: `['scale', 'clamp', 'log_sum_exp']`
- 失败的算子: `['matmul', 'mish', 'residual_add']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.matmul(x)` | — | ✗ | ⚠️ FAIL |
| `scale` | — | — | ✓ | ✅ PASS |
| `residual_add` | — | — | ✗ | ⚠️ FAIL |
| `clamp` | — | — | ✓ | ✅ PASS |
| `log_sum_exp` | — | — | ✓ | ✅ PASS |
| `mish` | — | — | ✗ | ⚠️ FAIL |

### matmul_scaling_residual_add.txt
- 待融合算子: `['matmul', 'scaling', 'residual_add']`
- 通过的算子: `['matmul']`
- 失败的算子: `['residual_add', 'scaling']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | — | — | ✓ | ✅ PASS |
| `scaling` | ✗ `return y * (1.0 + self.scaling_factor)` | — | ✗ | ⚠️ FAIL |
| `residual_add` | ✗ `return y * (1.0 + self.scaling_factor)` | — | ✗ | ⚠️ FAIL |

### matmul_sigmoid_sum.txt
- 待融合算子: `['matmul', 'sigmoid', 'sum']`
- 通过的算子: `['sigmoid', 'sum']`
- 失败的算子: `['matmul']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | — | ✗ `at::Tensor matmul_out = at::linear(x, weight, bias);` | ✗ | ⚠️ FAIL |
| `sigmoid` | — | — | ✓ | ✅ PASS |
| `sum` | — | — | ✓ | ✅ PASS |

### matmul_subtract_multiply_relu.txt
- 待融合算子: `['matmul', 'subtract', 'multiply', 'relu']`
- 通过的算子: `['subtract', 'multiply', 'relu']`
- 失败的算子: `['matmul']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | — | ✗ `at::Tensor linear_out = at::linear(x, weight, bias);` | ✗ | ⚠️ FAIL |
| `subtract` | — | — | ✓ | ✅ PASS |
| `multiply` | — | — | ✓ | ✅ PASS |
| `relu` | — | — | ✓ | ✅ PASS |

### matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp.txt
- 待融合算子: `['matmul', 'sum', 'max', 'avg_pool', 'log_sum_exp', 'log_sum_exp']`
- 通过的算子: `['avg_pool']`
- 失败的算子: `['log_sum_exp', 'matmul', 'max', 'sum']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | — | — | ✗ | ⚠️ FAIL |
| `sum` | ✗ `w_col_sum = self.linear.weight.sum(dim=0).contiguous()` | — | ✓ | ⚠️ FAIL |
| `max` | — | — | ✗ | ⚠️ FAIL |
| `avg_pool` | — | — | ✓ | ✅ PASS |
| `log_sum_exp` | — | — | ✗ | ⚠️ FAIL |
| `log_sum_exp` | — | — | ✗ | ⚠️ FAIL |

### matmul_swish_scaling.txt
- 待融合算子: `['matmul', 'swish', 'scaling']`
- 通过的算子: `['swish', 'scaling']`
- 失败的算子: `['matmul']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.matmul(x)` | — | ✗ | ⚠️ FAIL |
| `swish` | — | — | ✓ | ✅ PASS |
| `scaling` | — | — | ✓ | ✅ PASS |

### matmul_swish_sum_group_norm.txt
- 待融合算子: `['matmul', 'swish', 'sum', 'group_norm']`
- 通过的算子: `['sum', 'group_norm']`
- 失败的算子: `['matmul', 'swish']`

| token | A: forward 回退 | C: 绑定层 LibTorch | B: kernel 原语 | 结论 |
| --- | --- | --- | --- | --- |
| `matmul` | ✗ `x = self.matmul(x)` | — | ✗ | ⚠️ FAIL |
| `swish` | — | — | ✗ | ⚠️ FAIL |
| `sum` | — | — | ✓ | ✅ PASS |
| `group_norm` | — | — | — | ℹ️ 不可判定 |

## 完全通过的文件

- `conv2d_hard_swish_relu.txt`
- `gemm_add_relu.txt`
- `gemm_multiply_leakyrelu.txt`
- `gemm_sigmoid_scaling_residual_add.txt`
- `matmul_dropout_mean_softmax.txt`
