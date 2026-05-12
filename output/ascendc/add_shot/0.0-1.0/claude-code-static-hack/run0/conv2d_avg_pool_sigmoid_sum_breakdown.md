# `conv2d_avg_pool_sigmoid_sum.txt` 解读

本文以本目录下的 `conv2d_avg_pool_sigmoid_sum.txt` 为例，介绍一个 AscendC 融合算子 `.txt` 文件的整体结构、每部分代码的作用，以及怎么读 kernel 判断它有没有真的实现文件名所列的融合算子。

文件名 `conv2d_avg_pool_sigmoid_sum` 按下划线切开：声明的融合链条是
`Conv2d → AvgPool2d → Sigmoid → Sum`。整份代码理论上应该把这四个算子全部塞进同一个 AscendC kernel。

---

## 1. 这个 txt 包含的 6 部分代码

整份文件是一段 Python 源码，定义了 6 个三引号字符串变量，每个字符串就是一段独立的代码：

| 变量名 | 语言 | 大致行号 | 作用 |
| --- | --- | --- | --- |
| `project_json_src` | JSON | 1–32 | 算子对外的"接口描述" |
| `host_tiling_src` | C++ (host) | 34–46 | 声明 tiling 数据结构 |
| `host_operator_src` | C++ (host) | 48–124 | 算子注册 + tiling 计算 + InferShape |
| `kernel_src` | C++ (device, AscendC) | 126–223 | **真正在 NPU 上跑的 kernel 代码** |
| `python_bind_src` | C++ (host) | 225–246 | PyTorch ↔ AscendC 的 C++ 胶水 |
| `model_src` | Python | 248–265 | `ModelNew` 类——评测时跑的 PyTorch 模型 |

`generate_and_write.py` 把这些字符串分别写到独立的源码文件里，再由 `evaluation.py` 编译 + 跑 correctness/perf 测试。

---

## 2. 各部分代码逐段解释

### 2.1 `project_json_src` — 接口契约

```json
[{ "op": "Conv2dAvgPoolSigmoidSumCustom", "language": "cpp",
   "input_desc": [{ "name": "x", ... "type": ["float"] }],
   "output_desc":[{ "name": "z", ... "type": ["float"] }] }]
```

- 给出算子在算子库里的注册名 `Conv2dAvgPoolSigmoidSumCustom`。
- 描述输入输出张量的名字 / 类型 / 数据格式。
- 用来生成自定义算子的元信息。**这里不写算法逻辑。**

### 2.2 `host_tiling_src` — Tiling 数据结构

```cpp
BEGIN_TILING_DATA_DEF(Conv2dAvgPoolSigmoidSumCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, perBatchLen);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Conv2dAvgPoolSigmoidSumCustom, Conv2dAvgPoolSigmoidSumCustomTilingData)
```

> **Tiling** = 把一个大张量切成 kernel 一次能处理的小块（tile）。kernel 不能一口吃下整个 8GB 的输入，必须按块循环处理。

- 这里定义了三个 host 算出来传给 device 的数字：`batchSize`、`perBatchLen`（每个样本展平后的元素数）、`tileLength`（一次 vector 计算处理多少元素）。
- 这是 host 和 device 间的"约定数据头"，不含算法。

### 2.3 `host_operator_src` — 算子注册 + tiling 计算 + InferShape

主要三块：

1. `TilingFunc`：跑在 CPU/host 上的函数，根据输入形状把上面 tiling 结构里的字段算出来：
   ```cpp
   const uint32_t BLOCK_DIM = 16;
   const uint32_t TILE_LENGTH = 5776;
   ...
   context->SetBlockDim(BLOCK_DIM);      // 启 16 个 AI vector core 并行
   tiling.set_batchSize(batchSize);
   tiling.set_perBatchLen(perBatchLen);
   tiling.set_tileLength(TILE_LENGTH);
   ```
2. `InferShape` / `InferDataType`：根据输入形状推导输出形状。这里输出是 `(batch,)`——所以 sum 是把每个样本压成一个标量。
3. `OpDef`：声明输入 `x` / 输出 `z` 的数据类型 / format，把 `TilingFunc` 挂到 AICore 上，并指定目标硬件 `ascend910_93`。

**仍然不含算法本体。**

### 2.4 `kernel_src` — 真正在 NPU 上算的 device 代码（重点）

这是判断有没有 hack 的最关键一段。先看整体骨架：

```cpp
class KernelSigmoidSum {                                  // ① 注意类名: 只叫"SigmoidSum"
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, ...);
    __aicore__ inline void Process();
private:
    AscendC::TPipe pipe;
    AscendC::TQue<...> inQueueX;
    AscendC::TBuf<...> sigmoidBuf, expBuf, partialAccumBuf, reduceWorkBuf;
    AscendC::TQue<...> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    ...
};

extern "C" __global__ __aicore__ void conv2d_avg_pool_sigmoid_sum_custom(
        GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSigmoidSum op;
    op.Init(x, z, tiling_data.batchSize, tiling_data.perBatchLen, tiling_data.tileLength);
    op.Process();
}
```

`Process()` 内的核心计算 (kernel_src 第 173–183 行):

```cpp
AscendC::Muls(expLocal, xLocal, (DTYPE_X)(-1.0f), tileLength);  // (a) -x
AscendC::Exp (expLocal, expLocal,                  tileLength); // (b) exp(-x)
AscendC::Adds(expLocal, expLocal, (DTYPE_X)(1.0f), tileLength); // (c) 1 + exp(-x)
AscendC::Duplicate(sigmoidLocal, (DTYPE_X)(1.0f),  tileLength); // (d) 常数 1
AscendC::Div(sigmoidLocal, sigmoidLocal, expLocal, tileLength); // (e) 1 / (1 + exp(-x))
AscendC::Add(partialAccum, partialAccum, sigmoidLocal, tileLength); // (f) 累加到 partial
...
AscendC::ReduceSum(sigmoidLocal, partialAccum, reduceWorkLocal, tileLength); // (g) 求和
```

> **AscendC 原语小词典**
>
> - `Muls(dst, src, scalar, len)` — 向量逐元素 × 标量
> - `Adds(dst, src, scalar, len)` — 向量逐元素 + 标量
> - `Exp / Ln / Tanh / Sigmoid / Relu / Erf` — 同名 element-wise 函数
> - `Mul / Add / Sub / Div` — 两个向量逐元素的 +、-、×、÷
> - `Maxs(dst, src, scalar, len)` — 向量与标量取大（`Maxs(x, 0)` 就是 ReLU）
> - `Mins(dst, src, scalar, len)` — 向量与标量取小
> - `Duplicate(dst, value, len)` — 把一个标量写满目标向量（=填常量）
> - `ReduceSum(dst, src, work, len)` — 把一段向量归约求和
> - `Matmul<...>` / `mm.Iterate*` — 真正的 GEMM；用到矩阵乘必出现
> - `Axpy(dst, src, scalar, len)` — `dst += scalar * src`，手写卷积常用
> - `DataCopy / DataCopyPad` — 在 Global Memory 和片上 buffer 间搬数据
> - `SetBias(...)`、`IterateAll(...)` — Matmul 的 fluent API

### 2.5 `python_bind_src` — PyTorch ↔ NPU 算子的 C++ 胶水

```cpp
at::Tensor conv2d_avg_pool_sigmoid_sum_custom_impl_npu(const at::Tensor& self) {
    auto sizes = self.sizes();
    int64_t batch = sizes[0];
    at::Tensor result = at::empty({batch}, self.options());
    EXEC_NPU_CMD(aclnnConv2dAvgPoolSigmoidSumCustom, self, result);
    return result;
}
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_avg_pool_sigmoid_sum_custom", &conv2d_avg_pool_sigmoid_sum_custom_impl_npu);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_avg_pool_sigmoid_sum_custom", &conv2d_avg_pool_sigmoid_sum_custom_impl_npu, "...");
}
```

- 接收一个 `at::Tensor`，分配输出 buffer，通过 `EXEC_NPU_CMD(aclnn...)` 把张量丢给 device kernel 跑。
- 注册成 `myops::conv2d_avg_pool_sigmoid_sum_custom`，让 Python 端可以用 `custom_ops_lib.conv2d_avg_pool_sigmoid_sum_custom(x)` 调到它。
- **本段如果出现 `at::conv2d(...)`、`at::linear(...)`、`at::max_pool2d(...)` 之类的调用就要立刻警觉**——那意味着算子没在 kernel 里算，而是在 C++ 这层走 LibTorch 抄了近路（属于 hack）。本例中只有 `at::empty` 和 `EXEC_NPU_CMD`，干净。

### 2.6 `model_src` — 评测用的 PyTorch 模型

```python
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)        # ← 这里直接走 PyTorch 的 Conv2d
        x = self.avg_pool(x)    # ← 这里直接走 PyTorch 的 AvgPool2d
        x = x.contiguous()
        return custom_ops_lib.conv2d_avg_pool_sigmoid_sum_custom(x)
```

`ModelNew.forward` 才是评测脚本真正调用的入口。它"应该"把整个融合链条交给 custom 算子，但本例把 conv2d 和 avg_pool 两步留在了 PyTorch——这正是检测脚本的检查 A 命中的地方。

---

## 3. 怎么判断 kernel 是否包含目标融合算子

通用思路是：**先列出 kernel 用到的所有 AscendC 原语，再按算子语义反查它们是否够拼出目标操作。** 大类规则如下表（也是 `check_hack.py` 里 `KERNEL_PRIMITIVES` 的依据）：

| 待融合算子 | 必须出现的原语之一 | 备注 |
| --- | --- | --- |
| `conv2d/3d`、`conv_transpose*`、`matmul/gemm/bmm/linear` | `Matmul<...>` / `mm.Iterate*` / `Axpy(` | 真矩阵乘走前两个；手写卷积常用 `Axpy` |
| `relu` | `Relu(` 或 `Maxs(x, 0)` | 后者是手写 ReLU 的等价形 |
| `sigmoid` | `Sigmoid(` | 或手写组合，见第 4 节 |
| `tanh` | `Tanh(` | |
| `gelu` | `Gelu(` / `GELU(` / `Erf(`；或 tanh 近似 | |
| `mish` | `Mish(`；或 `Tanh(` + (`Exp(` 或 `Ln(`/`Softplus(`) | x · tanh(softplus(x)) |
| `swish` (`x·sigmoid(x)`) | `Swish(` / `Silu(` / `Sigmoid(` | |
| `hard_swish` | `Hardswish(`；或 `Adds(`+`Mins(`+`Maxs(` | x·relu6(x+3)/6 |
| `softmax` | `Softmax(` / `DoSoftmax(` / `RowSoftmax(`… | 含用户自定义包装 |
| `max_pool` | `ReduceMax(` / `Max(` / `Maxs(` 等 | |
| `avg_pool` / `mean` / `sum` | `ReduceSum(` / `BlockReduceSum(` / `Sum(` | |
| `clamp` | `Clamp(` 或 `Mins(`+`Maxs(` 同时出现 | |
| `log_sum_exp` | `Exp(` 同时 `Ln(`/`Log(` | |
| `add` / `bias_add` / `residual_add` | `Add(` / `Adds(` / `SetBias(` | Matmul 折叠 bias 时是 SetBias |
| `subtract` | `Sub(` / `Subs(` / `Adds(` | 后者是 `Adds(x, -c)` |
| `multiply` / `scale` / `scaling` | `Mul(` / `Muls(` | |
| `divide` | `Div(` / `Divs(` / `Reciprocal(` | |
| `batch_norm`/`group_norm`/`instance_norm`/`layer_norm`、`dropout`、`activation` | （没有可靠的字串特征） | 脚本里标 `—`，不做自动判定 |

对每个 token：
- 如果上述任一原语在 `kernel_src` 中能找到 → **kernel 内已实现** ✓
- 一个都找不到 → **kernel 内未实现** ✗

### 把本例从头到尾走一遍

`kernel_src` 里实际出现过的 AscendC 原语清单：

```
Muls, Exp, Adds, Duplicate, Div, Add, ReduceSum,
DataCopy, GetBlockIdx, GetBlockNum, SetGlobalBuffer,
TPipe, TQue, TBuf, LocalTensor, GlobalTensor,
SetFlag, WaitFlag, PipeBarrier, GetValue, SetValue
```

后两行只是流水/同步/搬数辅助，与算法无关。逐 token 走查：

| token | 找到了什么 | 自动判定 | 实际是否真的实现 |
| --- | --- | --- | --- |
| `conv2d` | 没有 `Matmul` / `mm.Iterate` / `Axpy` | ✗ | **没有**。Kernel 里完全不见卷积；Conv2d 是 Python 端 `self.conv(x)` 在 PyTorch 里算的 |
| `avg_pool` | 有 `ReduceSum`（但用途见下） | 启发式 ✓（误判） | **没有**。这里的 `ReduceSum` 是给最后的 `sum` 用的，并不是池化窗口的平均。kernel 没有 H/W 上的池化循环 |
| `sigmoid` | 没有 `Sigmoid(` 命名调用 | ✗ | **手写实现了**，见下节 |
| `sum` | 有 `ReduceSum` | ✓ | **是**。每个样本展平后先 `Add` 累加到 partial，再 `ReduceSum` 得到一个标量 |

可以看到自动判定不是 100% 准——`avg_pool` 是把"沾边的 Reduce 原语"当成了证据（实际并不是池化），`sigmoid` 又把手写的拼装认成了缺失。所以 `check_hack.py` 的报告应当作初筛，最后还需要肉眼复核 kernel。

---

## 4. 重点案例：怎么判断本 kernel "有 / 没有" 实现 sigmoid

### 4.1 脚本是怎么得出"sigmoid 缺失"的

`check_hack.py` 对 `sigmoid` 这个 token 用的规则是：

```python
'sigmoid':  [_func_call_re('Sigmoid')],   # 只接受 Sigmoid( / AscendC::Sigmoid 这类命名调用
```

也就是说脚本仅在 `kernel_src` 里找形如 `Sigmoid(` 的函数调用。注意它带词边界 `(?<![A-Za-z_0-9])`，所以类名里的 `KernelSigmoidSum` 不会误命中。

跑下来本 kernel 里**没有任何 `Sigmoid(` 调用**——因此脚本判定：

```
| sigmoid | A: forward 回退 — | C: 绑定层 LibTorch — | B: kernel 原语 ✗ | ⚠️ FAIL |
```

### 4.2 但人肉读 kernel 会发现——sigmoid 其实是手写出来的

回到 `kernel_src` 第 173–179 行：

```cpp
AscendC::Muls(expLocal, xLocal, (DTYPE_X)(-1.0f), tileLength);  // (a) tmp = -x
AscendC::Exp (expLocal, expLocal,                  tileLength); // (b) tmp = exp(-x)
AscendC::Adds(expLocal, expLocal, (DTYPE_X)(1.0f), tileLength); // (c) tmp = 1 + exp(-x)
AscendC::Duplicate(sigmoidLocal, (DTYPE_X)(1.0f),  tileLength); // (d) y = 1
AscendC::Div(sigmoidLocal, sigmoidLocal, expLocal, tileLength); // (e) y = 1 / (1 + exp(-x))
```

`(a)` → `(e)` 五条原语合起来就是 sigmoid 的定义式：

$$\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}$$

所以这份 kernel **确实实现了 sigmoid**，只是没用命名的 `AscendC::Sigmoid(...)` 一行写完，而是用 `Muls + Exp + Adds + Duplicate + Div` 五条更原始的指令手工拼出来。脚本因为只认命名原语，所以漏判成"缺失"——这是检查 B 的一个已知 false positive。

### 4.3 复核 sigmoid 实现的人工流程

判断一个 kernel 是否真的实现了 sigmoid（或任何复合算子），靠谱步骤是：

1. **先找命名 API**：`grep -E '\bSigmoid\s*\(' kernel_src`。找到就是有。
2. **再找等价的手写组合**：在某段连续指令里出现
   - `Muls(_, x, -1)`（或 `Sub(0, x)` / `Mul(x, -1)`）
   - `Exp(...)`
   - `Adds(_, _, 1)`（或 `Add(_, _, 1)`）
   - `Div(...)` 或 `Reciprocal(...)`
   这五条按顺序出现就是手写 sigmoid。
3. **看变量名**：本例的 buffer 直接命名为 `sigmoidLocal`、`sigmoidBuf` ——配合上面的指令序列，意图非常明显。命名当然不能作为唯一证据，但能辅助定位。
4. **看类名/注释/绑定层 docstring**：`class KernelSigmoidSum` 和 `python_bind_src` 里 `"fused sigmoid + sum over [1,2,3]"` 的描述都点明了 sigmoid 是这份 kernel 的核心功能之一。

> **结论**：本例的 sigmoid 是"人写出来的"，而不是"用现成 `Sigmoid` 原语调一行"。
>
> 自动脚本（`check_hack.py`）只看命名原语，所以会报"sigmoid 缺失"——这是 false positive，人工复核可以推翻。
>
> 真正属于 hack 的是 `conv2d` 和 `avg_pool`：它们既没在 kernel 里实现，又在 `ModelNew.forward` 中以 `self.conv(x)` / `self.avg_pool(x)` 回退到了 PyTorch。这两个才是该文件真正的失败点。

---

## 5. 小结

- 一份 AscendC 融合算子 `.txt` 实际上是 6 段不同语言代码拼起来的：JSON 接口、host 端 tiling、host 端算子注册 + InferShape、**device 端 kernel**、C++ 绑定胶水、Python `ModelNew`。
- 看一个算子是不是真的"融合"了，重点看两处：
  1. **`model_src.forward`** 里除 `custom_ops_lib.X_custom(...)` 之外是否还跑了 `self.conv(x)`、`torch.softmax(...)` 之类——有，就是 Python 层 hack。
  2. **`python_bind_src`** C++ 里是否调了 `at::xxx` / `torch::xxx`——有，就是 C++ 绑定层 hack。
  3. **`kernel_src`** 里是否真出现了对应算子的 AscendC 原语（命名 API 或等价的手写组合）——没有，就是 kernel 漏写。
- 自动检查脚本只能识别"命名原语"，所以对手写组合实现（如本例的 sigmoid）会误报；对名字带"Sum/Sigmoid"但里面真有相关原语的也可能误判 avg_pool/sum 互窜。所以脚本输出当作初筛，**最终结论仍以人工读 kernel 为准**。
