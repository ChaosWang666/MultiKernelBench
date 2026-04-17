# Ascend C API 最佳实践（完整版）

> Ascend C API 使用最佳实践。提供算术、归约、数据搬运、Buffer管理、精度转换等 API 的正确用法和限制说明。触发：用户询问具体 API 用法（如"DataCopy 怎么用"）、遇到 API 参数错误或限制报错（如 repeatTimes、对齐问题）、需要查看 API 最佳实践或避坑指南时。

---

## 目录

- [总览](#总览)
  - [API 类别索引](#api-类别索引)
  - [场景索引](#场景索引)
  - [API 黑名单](#api-黑名单)
- [API 快速参考](#api-快速参考)
- [算术运算 API 优化指南](#算术运算-api-优化指南)
- [Reduce API 使用指南](#reduce-api-使用指南)
- [Reduce Pattern 接口详解](#reduce-pattern-接口详解)
- [DataCopy / DataCopyPad 使用指南](#datacopy--datacopypad-使用指南)
- [UB 缓冲区管理指南](#ub-缓冲区管理指南)
- [精度转换与混合精度指南](#精度转换与混合精度指南)
- [流水线同步机制指南](#流水线同步机制指南)
- [Vector API repeatTime 参数限制](#vector-api-repeattime-参数限制)
- [API 使用限制与替代方案](#api-使用限制与替代方案)
- [Host 侧 Runtime API 使用规范](#host-侧-runtime-api-使用规范)

---

# 总览

## API 类别索引

| API 类别 | 涵盖 API | 核心文档 | 典型场景 |
|---------|---------|---------|---------|
| **算术运算** | Add, Sub, Mul, Div, Adds, Muls | [算术运算 API 优化指南](#算术运算-api-优化指南) | Softmax, LayerNorm, 广播优化 |
| **归约操作** | ReduceMax, ReduceSum | [Reduce API 使用指南](#reduce-api-使用指南), [Reduce Pattern 接口详解](#reduce-pattern-接口详解) | Softmax, LayerNorm, ReduceMean |
| **数据搬运** | DataCopy, DataCopyPad | [DataCopy / DataCopyPad 使用指南](#datacopy--datacopypad-使用指南) | 非对齐处理、多维搬运 |
| **Buffer 管理** | TBuf, TQue | [UB 缓冲区管理指南](#ub-缓冲区管理指南) | Double Buffer、内存规划 |
| **精度转换** | Cast | [精度转换与混合精度指南](#精度转换与混合精度指南) | FP16/FP32 混合精度 |
| **流水线同步** | EnQue, DeQue, SetFlag | [流水线同步机制指南](#流水线同步机制指南) | 多级流水线、事件同步 |
| **Compare 256B对齐** | Compare | [API 使用限制与替代方案 §2.1](#21-compare-api-256字节对齐约束) | Padding 策略 |
| **repeatTime 限制** | repeatTimes ≤ 255 | [Vector API repeatTime 参数限制](#vector-api-repeattime-参数限制) | 分批处理 |
| **API 限制** | - | [API 使用限制与替代方案](#api-使用限制与替代方案) | 禁用 API、编译期限制 |
| **Host Runtime** | aclrtSetDevice, aclrtGetDeviceInfo | [Host 侧 Runtime API 使用规范](#host-侧-runtime-api-使用规范) | 设备初始化、核数获取 |

## 场景索引

| 使用场景 | 相关文档 | 关键技巧 |
|---------|---------|---------|
| **Softmax/LayerNorm** | Reduce / Reduce-Pattern / 算术运算 | 标量操作、广播优化、Buffer 复用 |
| **逐行处理（AR 模板）** | 算术运算 | Adds/Muls、节省 UB |
| **多行广播（ARA 模板）** | 算术运算 | BinaryRepeatParams.src1RepStride=0、分批处理 |
| **非对齐数据** | DataCopy | DataCopyPad、32 字节对齐 |
| **混合精度** | 精度转换 | FP16 输入 FP32 计算 |
| **流水线优化** | 流水线 / Buffer | Double Buffer、事件同步 |
| **性能调优** | Buffer / repeat-limits | Double Buffer、repeatTimes 优化 |
| **遇到 API 限制** | API 限制 | 替代方案、避坑指南 |

## API 黑名单

**禁止在生产代码中使用**：

| API | 禁止原因 | 替代方案 |
|-----|---------|---------|
| `GlobalTensor::SetValue()` | 效率极低 | `DataCopyPad` |
| `GlobalTensor::GetValue()` | 效率极低 | `DataCopyPad` |

**限制使用的 API**：

| API | 限制条件 | 说明 |
|-----|---------|------|
| `DataCopy(GM↔UB)` | 仅当搬运数据**严格 32 字节对齐**时允许使用 | 非对齐场景必须使用 `DataCopyPad` |

**仅允许调试时使用**：
```cpp
// ✅ 调试：单点验证
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));
```

---

# API 快速参考

Ascend C API 使用核心决策索引。

## 核心原则速查

### 1. DataCopy vs DataCopyPad

**原则：优先使用 DataCopyPad**

| 场景 | API | 原因 |
|-----|-----|------|
| 所有 GM ↔ UB 搬运 | `DataCopyPad` | 统一处理对齐/非对齐 |
| 确定数据严格 32 字节对齐 | `DataCopy` | 简单场景可用 |

### 2. Cast RoundMode 选择

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| half → float | `CAST_NONE` | 低→高精度，无损失 |
| float → half | `CAST_ROUND` | 高→低精度，需舍入 |
| half → int32_t | `CAST_ROUND` | 量化场景 |
| int32_t → float | `CAST_NONE` | 整数→浮点 |

### 3. TBuf vs TQue 选择

| 场景 | 类型 | 说明 |
|------|------|------|
| MTE2/MTE3 搬运缓冲区 | `TQue<VECIN/VECOUT>` | `InitBuffer(que, num, size)` |
| 纯 Vector 计算缓冲区 | `TBuf<VECCALC>` | `InitBuffer(buf, size)` |

### 4. 流水线同步

**原则**：必须用 EnQue/DeQue 同步 MTE 和 Vector

**核心模式**：
```
CopyIn → EnQue → DeQue → Compute → EnQue → DeQue → CopyOut
```

### 5. Vector API repeatTime 限制

**核心限制**：repeatTime 为 uint8_t 时，最大值 255

**处理方法**：Host 侧限制 R_max 或 Kernel 侧分批处理

### 6. Reduce API 选择

| 场景 | 接口 | 说明 |
|-----|------|------|
| 逐行独立 Reduce | Level 2: `ReduceMax(dst, src, tmp, count)` | 无对齐要求，count 传 rLength |
| 跨行批量 Reduce | Pattern: `ReduceMax<T, Pattern::AR>(...)` | 需 32 字节对齐 |

## 决策树：我应该用什么API？

### Q1: 需要 GM ↔ UB 搬运数据？

```
是 → DataCopyPad（推荐）
   → DataCopy（仅当确定32字节对齐时）

否 → 继续
```

### Q2: 需要精度转换？

```
half → float → CAST_NONE
float → half → CAST_ROUND
其他 → 查阅 Cast RoundMode 表
```

### Q3: 需要分配 UB 缓冲区？

```
涉及 MTE 搬运 → TQue + InitBuffer(que, num, size)
纯 Vector 计算 → TBuf + InitBuffer(buf, size)
```

### Q4: 遇到数据错误/随机值？

```
1. 检查是否缺少 EnQue/DeQue
2. 检查 DataCopyPad 参数
3. 检查 Reduce API 的 tmpBuffer 类型
4. 检查多行处理时的 rowOffset 计算
5. 检查 repeatTime 是否溢出
```

### Q5: 需要混合精度计算（FP16 输入，FP32 中间计算）？

```
查阅混合精度模式节
```

---

# 算术运算 API 优化指南

> **适用场景**：使用算术运算 API（Add/Sub/Mul/Div）时，选择最优实现方式，避免不必要的广播 buffer 和指令开销。

## 概述

算术运算 API（Add/Sub/Mul/Div）支持两种使用模式：

| 模式 | API | 适用场景 | Buffer 需求 |
|-----|-----|---------|------------|
| **标量操作** | `Adds/Muls` | 单行处理（Softmax AR 模板） | 32B |
| **广播操作** | `Sub/Div + BinaryRepeatParams` | 多行处理（Softmax ARA 模板） | alignedCols×4 |

**关键优化**：
- 单行：使用 `Adds/Muls` 避免 Duplicate
- 多行：使用 `src1RepStride=0` 避免逐行循环

## 场景1：标量操作（单行）

### 方案对比

**问题**：需要对 tensor 每个元素执行 `x - scalar` 或 `x / scalar`

**典型场景**：
- Softmax AR 模板：`x - max_val`（数值稳定）
- Softmax AR 模板：`exp(x) / sum`（归一化）
- LayerNorm：`x - mean`（中心化）
- BatchNorm：`x * gamma + beta`

**方案对比**：

| 方案 | 指令数 | Buffer 需求 | 推荐度 |
|-----|--------|------------|--------|
| Duplicate + Sub | 2 条 | `rLength × sizeof(T)` | ⭐⭐ |
| Duplicate + Div | 2 条 | `rLength × sizeof(T)` | ⭐⭐ |
| **Adds(-scalar)** | **1 条** | **32B** | **⭐⭐⭐⭐⭐** |
| **Muls(1/scalar)** | **1 条** | **32B** | **⭐⭐⭐⭐⭐** |

### API 接口

**Adds（标量加法）**：
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src,
    const T& scalarValue,
    const int32_t& count);

// 功能: dst[i] = src[i] + scalarValue
// 示例: Adds(dst, src, -maxVal, count)  // 减法转加法
```

**Muls（标量乘法）**：
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src,
    const T& scalarValue,
    const int32_t& count);

// 功能: dst[i] = src[i] * scalarValue
// 示例: Muls(dst, src, 1.0/sum, count)  // 除法转乘法
```

### 完整示例

#### 优化前（Sub/Div + Duplicate）

```cpp
// Buffer 初始化
uint32_t broadcastBufSize = rLengthAlign * sizeof(T);  // 例如：512B (rLength=128, FP32)
pipe.InitBuffer(broadcastBuf, broadcastBufSize);
pipe.InitBuffer(reduceBuf, reduceBufSize);

// Compute
LocalTensor<T> broadcastLocal = broadcastBuf.Get<T>();

for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;

    // Step 1: ReduceMax
    ReduceMax<T>(broadcastLocal, xLocal[rowOffset], reduceTmpLocal, rLength, false);

    // Step 2: Duplicate + Sub（需要广播 buffer）
    T maxVal = broadcastLocal.GetValue(0);
    Duplicate<T>(broadcastLocal, maxVal, rLength);  // 指令 1
    Sub<T>(yLocal[rowOffset], xLocal[rowOffset], broadcastLocal, rLength);  // 指令 2

    // Step 3: Exp
    Exp<T>(yLocal[rowOffset], yLocal[rowOffset], rLength);

    // Step 4: ReduceSum
    ReduceSum<T, true>(broadcastLocal, yLocal[rowOffset], reduceTmpLocal, rLength);

    // Step 5: Duplicate + Div（需要广播 buffer）
    T sumVal = broadcastLocal.GetValue(0);
    Duplicate<T>(broadcastLocal, sumVal, rLength);  // 指令 3
    Div<T>(yLocal[rowOffset], yLocal[rowOffset], broadcastLocal, rLength);  // 指令 4
}

// 总计：6 条指令/行，需要 broadcastBuf (512B for rLength=128)
```

#### 优化后（Adds/Muls + 标量）

```cpp
// Buffer 初始化（节省 broadcastBuf）
uint32_t scalarBufSize = 32;  // 最小对齐要求，仅需存储 1 个标量
pipe.InitBuffer(scalarBuf, scalarBufSize);
pipe.InitBuffer(reduceBuf, reduceBufSize);

// Compute
LocalTensor<T> scalarLocal = scalarBuf.Get<T>();

for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;

    // Step 1: ReduceMax
    ReduceMax<T>(scalarLocal, xLocal[rowOffset], reduceTmpLocal, rLength, false);

    // Step 2: Adds（直接标量操作，无需广播）
    T maxVal = scalarLocal.GetValue(0);
    Adds<T>(yLocal[rowOffset], xLocal[rowOffset], -maxVal, rLength);  // 指令 1

    // Step 3: Exp
    Exp<T>(yLocal[rowOffset], yLocal[rowOffset], rLength);

    // Step 4: ReduceSum
    ReduceSum<T, true>(scalarLocal, yLocal[rowOffset], reduceTmpLocal, rLength);

    // Step 5: Muls（除法转乘法，直接标量操作）
    T sumVal = scalarLocal.GetValue(0);
    T invSumVal = (T)1.0 / sumVal;  // CPU 端计算 1/sum
    Muls<T>(yLocal[rowOffset], yLocal[rowOffset], invSumVal, rLength);  // 指令 2
}

// 总计：4 条指令/行，节省 broadcastBuf (480B for rLength=128)
```

## 场景2：广播操作（多行）

### 方案对比

**问题**：需要对多行数据执行相同的标量操作（如 `x - max`、`exp / sum`）

**方案对比**：

| 方案 | API 调用 | Buffer 需求 | 推荐度 |
|-----|---------|------------|--------|
| 逐行循环 | R 次 | alignedCols×4 | ⭐⭐ |
| 单次广播（R ≤ 64） | 1 次 | alignedCols×4 | ⭐⭐⭐⭐⭐ |
| 分批广播（R > 64） | ceil(R/64) 次 | alignedCols×4 | ⭐⭐⭐⭐⭐ |

### 核心原理

**BinaryRepeatParams.src1RepStride=0 实现广播**：

```cpp
struct BinaryRepeatParams {
    uint8_t dstBlkStride;    // 单次迭代内，dst 的 block 步长
    uint8_t src0BlkStride;   // 单次迭代内，src0 的 block 步长
    uint8_t src1BlkStride;   // 单次迭代内，src1 的 block 步长
    uint8_t dstRepStride;    // 相邻迭代间，dst 的 block 步长
    uint8_t src0RepStride;   // 相邻迭代间，src0 的 block 步长
    uint8_t src1RepStride;   // =0 实现广播
};
```

**工作原理**：
- `dstRepStride = alignedCols/8`：每次迭代，dst 前进 `alignedCols` 个元素
- `src0RepStride = alignedCols/8`：每次迭代，src0 前进 `alignedCols` 个元素
- `src1RepStride = 0`：每次迭代，src1 **不前进**，重复读取相同位置

**效果**：
```
迭代 0: dst[0:cols]     = src0[0:cols]     - src1[0:cols]
迭代 1: dst[cols:2cols] = src0[cols:2cols] - src1[0:cols]  ← 重复读取
迭代 2: dst[2cols:3cols]= src0[2cols:3cols]- src1[0:cols]  ← 重复读取
```

### 分批处理

#### 方案1：逐行循环（低效）

```cpp
for (uint32_t r = 0; r < R; r++) {
    Sub(dstLocal[r * alignedCols], srcLocal[r * alignedCols], scalarLocal, alignedCols);
}
// API 调用：R 次
```

#### 方案2：单次广播（高效，R ≤ 64）

```cpp
uint64_t mask = alignedCols;
uint8_t repeatTime = R;

Sub(dstLocal, srcLocal, scalarLocal, mask, repeatTime,
    {1, 1, 1, alignedCols/8, alignedCols/8, 0});
// API 调用：1 次
// 性能提升：R 倍
```

#### 方案3：分批广播（高效，R > 64）

```cpp
constexpr uint32_t BATCH_SIZE = 64;
uint32_t totalBatches = (R + BATCH_SIZE - 1) / BATCH_SIZE;  // ceil(R/64)

for (uint32_t batch = 0; batch < totalBatches; batch++) {
    uint32_t startRow = batch * BATCH_SIZE;
    uint8_t repeatTime = (startRow + BATCH_SIZE <= R) ? BATCH_SIZE : (R - startRow);
    uint32_t offset = startRow * alignedCols;

    Sub(dstLocal[offset], srcLocal[offset], scalarLocal,
        mask, repeatTime, {1, 1, 1, alignedCols/8, alignedCols/8, 0});
}
// API 调用：ceil(R/64) 次
// 性能提升：约 64 倍
```

## 性能对比

### 标量操作（单行）

| 项目 | 优化前 | 优化后 | 改善 |
|-----|--------|--------|------|
| **指令数/行** | 6 条 | 4 条 | **-33%** |
| **Buffer 大小** | 512B (rLength=128) | 32B | **-94%** |
| **UB 节省** | - | ~480B | 可用于更大 rowsPerLoop |

### 广播操作（多行）

| R (行数) | 逐行循环 | 单次广播 | 分批广播 | 性能提升 |
|---------|---------|---------|---------|---------|
| 32 | 32 次 | 1 次 | - | **32×** |
| 64 | 64 次 | 1 次 | - | **64×** |
| 100 | 100 次 | - | 2 次 | **50×** |
| 128 | 128 次 | - | 2 次 | **64×** |
| 200 | 200 次 | - | 4 次 | **50×** |

### 实测示例（Softmax ARA 分支）

**场景**：R=128, alignedCols=64, FP32

| 操作 | 优化前 | 优化后 | 提升 |
|-----|--------|--------|------|
| Sub (x-max) | 128 次 | 2 次 | 64× |
| Div (exp/sum) | 128 次 | 2 次 | 64× |
| **总计** | **256 次** | **4 次** | **64×** |

## 适用 API

所有支持 `BinaryRepeatParams` 的二元运算 API：

| API | 用途 | 单行优化 | 多行优化 |
|-----|------|---------|---------|
| **Add** | 加法 | Adds | src1RepStride=0 |
| **Sub** | 减法 | Adds(-val) | src1RepStride=0 |
| **Mul** | 乘法 | Muls | src1RepStride=0 |
| **Div** | 除法 | Muls(1/val) | src1RepStride=0 |
| **Max** | 最大值 | - | src1RepStride=0 |
| **Min** | 最小值 | - | src1RepStride=0 |

## 常见错误

| 错误 | 原因 | 解决方案 |
|-----|------|---------|
| 编译错误：mask 超限 | `mask > 64` (FP32) | 分批处理或回退循环 |
| 数据错误 | `src1RepStride` 未设置为 0 | 确认参数：`{..., 0}` |
| 部分行正确 | offset 计算错误 | `offset = startRow * alignedCols` |
| 越界崩溃 | repeatTime 计算错误 | 使用三目运算 |
| Buffer 不足 | 使用 Duplicate 方案 | 改用 Adds/Muls |
| dst == tmpBuffer | Reduce API 限制 | 使用不同 buffer |

## 检查清单

**标量操作（单行）**：
- [ ] 使用 `Adds(-scalar)` 替代 `Duplicate + Sub`
- [ ] 使用 `Muls(1/scalar)` 替代 `Duplicate + Div`
- [ ] 标量除法转换为乘法（CPU 端计算 1/scalar）

**广播操作（多行）**：
- [ ] alignedCols ≤ 64 (FP32) / ≤ 128 (FP16)
- [ ] 使用 `src1RepStride = 0` 实现广播
- [ ] R > 64 时使用分批处理
- [ ] offset 计算正确：`offset = startRow * alignedCols`

---

# Reduce API 使用指南

逐行 Reduce 与跨行 Reduce 的 API 选择与使用规范。

## 接口选择

| 场景 | 接口 | 参数 | 对齐要求 | 典型用途 |
|-----|------|------|----------|---------|
| 逐行独立处理 | Level 2 | `(dst, src, tmp, count)` | **无** | Softmax, LayerNorm |
| 跨行批量处理 | Pattern | **两种形式**（见下文） | 32 字节 | ReduceSum axis=-1 |

**选择原则**：
- 逐行独立计算 → **Level 2 接口**（更简单，无对齐要求）
- 需要跨行 Reduce → **Pattern 接口**（性能更高，推荐形式1）

## Level 2 接口（逐行处理）

### API 签名

```cpp
AscendC::ReduceMax<T>(dst, src, tmpBuffer, count, calIndex);
AscendC::ReduceSum<T, isSetMask=true>(dst, src, tmpBuffer, count);
AscendC::ReduceMin<T>(dst, src, tmpBuffer, count, calIndex);
```

**参数**：
- `dst`：输出 LocalTensor（1 个元素）
- `src`：输入 LocalTensor（count 个元素）
- `tmpBuffer`：临时 buffer（**类型必须与 T 相同**）
- `count`：元素个数（int32_t）
- `calIndex`：是否计算索引（bool，默认 false）

### tmpBuffer 类型要求

**tmpBuffer 类型必须与 dst/src 相同**：

```cpp
// ❌ 错误：tmpBuffer 类型不匹配
AscendC::LocalTensor<uint8_t> tmpBuffer = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax(rowTmp, src, tmpBuffer, count);  // 编译错误！

// ✅ 正确：tmpBuffer 类型必须与 T 相同
AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();
AscendC::ReduceMax(rowTmp, src, reduceTmp, count);
```

### 完整示例：Softmax 逐行处理

```cpp
__aicore__ inline void ProcessRow(
    AscendC::LocalTensor<T>& xLocal,
    AscendC::LocalTensor<T>& yLocal,
    uint32_t rowIdx)
{
    uint32_t rowOffset = rowIdx * rLengthAlign;  // ⚠️ 用 rLengthAlign

    AscendC::LocalTensor<T> rowTmp = rowBuf.Get<T>();
    AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();

    // 1. ReduceMax（count = rLength，有效数据个数）
    AscendC::ReduceMax<T>(rowTmp, xLocal[rowOffset], reduceTmp,
        static_cast<int32_t>(rLength), false);

    T maxVal = rowTmp.GetValue(0);
    AscendC::Duplicate<T>(rowTmp, maxVal, rLength);
    AscendC::Sub<T>(xLocal[rowOffset], xLocal[rowOffset], rowTmp, rLength);

    // 2. Exp
    AscendC::Exp<T>(xLocal[rowOffset], xLocal[rowOffset], rLength);

    // 3. ReduceSum
    AscendC::ReduceSum<T, true>(rowTmp, xLocal[rowOffset], reduceTmp,
        static_cast<int32_t>(rLength));

    T sumVal = rowTmp.GetValue(0);
    AscendC::Duplicate<T>(rowTmp, sumVal, rLength);
    AscendC::Div<T>(yLocal[rowOffset], xLocal[rowOffset], rowTmp, rLength);
}
```

## Pattern 接口（跨行批量）

Pattern 接口有**两种重载形式**，详见 [Reduce Pattern 接口详解](#reduce-pattern-接口详解)。

### 快速入门

```cpp
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();
AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

uint32_t srcShape[] = {rows, alignedCols};  // alignedCols 必须 32 字节对齐

// 推荐使用形式1：显式传入 tmpLocal
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);
```

### 关键要点

| 要点 | 说明 |
|-----|------|
| **对齐要求** | `alignedCols` 必须 32 字节对齐 |
| **Pattern 类型** | `Pattern::Reduce::AR`（沿列方向）、`Pattern::Reduce::RA`（沿行方向） |
| **推荐形式** | 形式1（显式传入 sharedTmpBuffer） |
| **临时空间** | 两种形式都需要预留 |

### 非对齐数据处理

```cpp
// ✅ 方案1：改用 Level 2 接口（无对齐要求）
AscendC::ReduceMax<T>(dst, src, tmp, rLength, false);

// ✅ 方案2：用 DataCopyPad 填充到对齐
uint32_t alignedCols = ((rLength * sizeof(T) + 31) / 32) * 32 / sizeof(T);
AscendC::DataCopyPadExtParams<T> padParams;
padParams.isPad = true;
padParams.rightPadding = alignedCols - rLength;
DataCopyPad(dstLocal, srcGm, copyParams, padParams);

uint32_t srcShape[] = {1, alignedCols};
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, true);
```

## 常见错误

### 错误1：tmpBuffer 类型不匹配

```cpp
// ❌ 错误
AscendC::LocalTensor<uint8_t> tmpBuffer = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax(rowTmp, src, tmpBuffer, count);

// ✅ 正确
AscendC::LocalTensor<T> reduceTmp = reduceBuf.Get<T>();
AscendC::ReduceMax(rowTmp, src, reduceTmp, count);
```

### 错误2：rowOffset 用 rLength 而非 rLengthAlign

```cpp
// ❌ 错误：单行通过，多行失败
uint32_t rowOffset = rowIdx * rLength;

// ✅ 正确
uint32_t rowOffset = rowIdx * rLengthAlign;
```

### 错误3：非对齐数据用 Pattern 接口

```cpp
// ❌ 错误：rLength=13，非 32 字节对齐
uint32_t srcShape[] = {1, rLength};
AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, false);

// ✅ 方案1：改用 Level 2 接口
AscendC::ReduceMax<T>(dst, src, tmp, rLength, false);

// ✅ 方案2：用 DataCopyPad 填充到对齐（见上文）
```

### 错误4：Reduce API count 传 rLengthAlign

```cpp
// ❌ 错误：count 应该是有效数据个数
AscendC::ReduceMax(rowTmp, src, tmp, rLengthAlign, false);

// ✅ 正确：count 只传有效数据个数
AscendC::ReduceMax(rowTmp, src, tmp, rLength, false);
```

### 错误5：Pattern 接口形式2 忘记预留临时空间

```cpp
// ❌ 错误：运行时 UB 越界或结果错误
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, true);

// ✅ 方案1：使用形式1（推荐）
AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(dst, src, tmpLocal, srcShape, true);
```

## 最佳实践

### 参数对照表

| 参数位置 | 用 rLength | 用 rLengthAlign |
|---------|-----------|-----------------|
| DataCopyPad blockLen | ✓ | ✗ |
| Reduce API count | ✓ | ✗ |
| Sub/Exp/Div count | ✓ | ✗ |
| UB rowOffset | ✗ | ✓ |
| Buffer 大小计算 | ✗ | ✓ |

### 决策流程

```
需要 Reduce 操作？
    │
    ├─ 逐行独立处理（Softmax/LayerNorm）
    │     └─→ Level 2 接口
    │           - 无对齐要求
    │           - count = rLength
    │
    └─ 跨行批量 Reduce
          └─→ Pattern 接口（形式1 推荐）
                - 需要 32 字节对齐
                - 显式管理 tmp buffer
```

### Buffer 分配

```cpp
uint32_t tileSize = rowsPerLoop * rLengthAlign * sizeof(T);
uint32_t rowBufSize = rLengthAlign * sizeof(T);
uint32_t reduceBufSize = 32 * 1024;

pipe->InitBuffer(inQueueX, 1, tileSize);
pipe->InitBuffer(outQueueY, 1, tileSize);
pipe->InitBuffer(rowBuf, rowBufSize);
pipe->InitBuffer(reduceBuf, reduceBufSize);
```

---

# Reduce Pattern 接口详解

跨行批量 Reduce 的 Pattern 接口高级用法。

## 两种重载形式

### 形式1：显式传入 sharedTmpBuffer（推荐）

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(
    const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer,  // 显式传入
    const uint32_t srcShape[],
    bool srcInnerPad
);
```

### 形式2：框架自动申请临时空间

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(
    const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& srcTensor,
    const uint32_t srcShape[],
    bool srcInnerPad
);
```

> **⚠️ 形式2 必须预留临时空间**，否则运行时 UB 越界。详见下文临时空间预留。

## Pattern 类型

| Pattern | 方向 | 输入形状 | 输出形状 | 用途 |
|---------|-----|---------|---------|------|
| `Pattern::Reduce::AR` | 沿最后一维（列方向） | (R, C) | (R,) | 每行归约为1个值 |
| `Pattern::Reduce::RA` | 沿第一维（行方向） | (R, C) | (C,) | 每列归约为1个值 |

## 参数说明

| 参数 | 类型 | 说明 |
|-----|------|------|
| `T` | half/float | 数据类型 |
| `pattern` | Pattern::Reduce::AR/RA | 归约模式 |
| `isReuseSource` | bool | 是否复用源操作数（默认 false） |
| `dstTensor` | LocalTensor\<T\> | 输出张量 |
| `srcTensor` | LocalTensor\<T\> | 输入张量 |
| `sharedTmpBuffer` | LocalTensor\<uint8_t\> | 临时缓存（形式1） |
| `srcShape` | uint32_t[] | `{rows, alignedCols}`，**alignedCols 必须 32 字节对齐** |
| `srcInnerPad` | bool | A2/A3 芯片只支持 `true` |

## 临时空间预留

**两种形式都需要预留临时空间**：

| 方式 | 预留方法 | 优点 | 推荐度 |
|-----|---------|------|-------|
| **形式1** | `InitBuffer(tmpBuf, tmpSize)` + 显式传入 | 内存可控、可复用 | ⭐⭐⭐⭐⭐ |
| **形式2** | `InitBuffer(tmpBuf, tmpSize)`（框架自动使用） | 代码简洁 | ⭐⭐⭐ |

**临时空间大小计算**：

```cpp
#include "kernel_operator.h"

uint32_t maxSize, minSize;
AscendC::GetReduceMaxMaxMinTmpSize(srcShape, sizeof(T), isReuse, maxSize, minSize);

// 使用 maxSize（安全）或 minSize（节省内存）
pipe->InitBuffer(tmpBuf, maxSize);
```

## 完整示例

### 示例1：ReduceMax（AR/RA Pattern）

```cpp
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();
AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

uint32_t srcShape[] = {rows, alignedCols};  // alignedCols 必须 32 字节对齐
constexpr bool isReuse = true;

// AR Pattern：每行归约为1个值 → 输出 rows 个值
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, isReuse>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);

// RA Pattern：每列归约为1个值 → 输出 alignedCols 个值
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, isReuse>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);
```

### 示例2：ReduceSum（框架自动申请）

```cpp
// ⚠️ 必须提前预留临时空间
AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();

uint32_t srcShape[] = {rows, alignedCols};

AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
    dstLocal, srcLocal, srcShape, true);
```

## 对比总结

| 对比项 | 形式1（显式传入） | 形式2（框架申请） |
|-------|-----------------|------------------|
| tmp 参数 | ✅ 显式传入 | ❌ 框架自动申请 |
| 预留空间 | ✅ 调用时传入即可 | ⚠️ **必须在 InitBuffer 预留** |
| 内存管理 | 手动管理，可复用 | 需提前预留，易遗漏 |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**推荐使用形式1**，避免遗漏预留空间导致运行时错误。

---

# DataCopy / DataCopyPad 使用指南

GM ↔ UB 数据搬运的完整指南。

## 选择规则

**原则：优先使用 DataCopyPad**

| 场景 | API | 原因 |
|-----|-----|------|
| **非对齐或不确定对齐** | `DataCopyPad` | 自动处理对齐/非对齐，避免边界 bug |
| **数据量严格 32 字节对齐** | `DataCopy` 或 `DataCopyPad` | 确定对齐时 DataCopy 可用，DataCopyPad 更安全 |

### ⛔️ 黑名单 API（禁止在生产代码中使用）

| API | 禁止原因 | 仅允许场景 |
|-----|---------|-----------|
| `GlobalTensor::SetValue(idx, val)` | 效率极低，单元素逐个写入 | **仅调试时使用** |
| `GlobalTensor::GetValue(idx)` | 效率极低，单元素逐个读取 | **仅调试时使用** |

```cpp
// ❌ 禁止：生产代码使用 SetValue/GetValue
for (uint32_t i = 0; i < size; i++) {
    xGm.SetValue(i, value);    // ⛔️ 效率极低
    T val = xGm.GetValue(i);   // ⛔️ 效率极低
}

// ✅ 正确：使用 DataCopyPad 批量搬运
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);

// ✅ 允许：调试时单点验证
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));  // 仅调试
```

**为什么优先 DataCopyPad？**

1. 自动处理非对齐，无需手动判断
2. CopyIn 和 CopyOut 都适用
3. Tiling 设计时可能产生非对齐的 tile 大小
4. 对齐场景下性能差异可忽略

## 32 字节对齐要求

**DataCopy 要求 32 字节对齐**，非对齐会导致数据错误。

| 数据类型 | 对齐元素数 | 最小对齐字节数 |
|---------|-----------|--------------|
| half (2 bytes) | 16 | 32 |
| float (4 bytes) | 8 | 32 |
| int32_t (4 bytes) | 8 | 32 |

## DataCopyPad 参数详解

### isPad 参数

| isPad | 含义 |
|-------|------|
| `false` | 框架自动填充，用户不指定填充值 |
| `true` | 用用户指定的 `paddingValue` 填充 |

### blockLen 非对齐时的填充行为

**GM → UB（CopyIn）**：

| 条件 | isPad | dummy 填充值 |
|-----|-------|-------------|
| leftPadding=0, rightPadding=0 | false | **第一个元素值** |
| leftPadding=0, rightPadding=0 | true | paddingValue |
| leftPadding≠0 或 rightPadding≠0 | false | 随机值 |
| leftPadding≠0 或 rightPadding≠0 | true | paddingValue |

**UB → GM（CopyOut）**：
- 框架自动处理非对齐
- 搬到 GM 时自动丢弃 dummy

## 使用场景示例

### 场景1：非对齐 CopyIn，不关心填充值

```cpp
// cols=5 (FP32)，blockLen=20字节，非对齐
// 后续计算只处理 cols 个元素，dummy 被忽略
AscendC::DataCopyParams copyParams{1, cols * sizeof(float), 0, 0};
AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);

// 后续计算只处理 cols 个元素
AscendC::ReduceMax(tmpReduce, xLocal, tmpReduce, cols, false);
```

### 场景2：非对齐 CopyIn，指定填充值

```cpp
uint32_t padElements = paddedCols - cols;
AscendC::DataCopyPadExtParams<float> padParams{true, 0, padElements, 0.0f};
AscendC::DataCopyExtParams copyParams{1, cols * sizeof(float), 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

### 场景3：非对齐 CopyOut

```cpp
// CopyOut 自动处理非对齐，搬到 GM 时丢弃 dummy
AscendC::DataCopyParams copyParams{1, cols * sizeof(float), 0, 0};
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 完整示例：多行批量搬运

```cpp
__aicore__ inline void CopyInBatch(uint32_t startLocalRow, uint32_t rowsThisTile)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = rowsThisTile;
    copyParams.blockLen = cols * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = paddedColsT - cols;
    padParams.paddingValue = 0;

    AscendC::DataCopyPad(xLocal, xGm[startLocalRow * cols], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}
```

### 场景4：逐行独立处理模式（Softmax/LayerNorm 推荐）

**适用场景**：Softmax / LayerNorm 等逐行独立计算的算子，不需要跨行 Reduce。

**核心要点**：blockCount 模式 + UB 对齐存储

```cpp
// ========== Tiling 参数 ==========
uint32_t rLength = 13;                           // 有效数据个数
uint32_t rLengthAlign = (rLength + 7) / 8 * 8;   // 对齐到 8 元素（FP32 下 32 字节）

// ========== 数据搬运 ==========
AscendC::DataCopyPad(xLocal, xGm[offset],
    {static_cast<uint16_t>(rows),           // blockCount: 行数
     static_cast<uint32_t>(rLength * sizeof(T)), // blockLen: 有效数据长度（非对齐！）
     0, 0},                                  // stride: 连续存储
    {false, 0, 0, 0});                       // padParams: 自动处理

inQueueX.EnQue(xLocal);
auto xIn = inQueueX.DeQue<T>();

// ========== 逐行处理 ==========
for (uint32_t row = 0; row < rows; row++) {
    // 关键：UB 偏移用 rLengthAlign，不是 rLength！
    uint32_t rowOffset = row * rLengthAlign;

    // Reduce API 只传 rLength（有效数据个数）
    AscendC::ReduceMax<T>(rowTmp, xIn[rowOffset], reduceTmp,
        static_cast<int32_t>(rLength), false);
    // ... Sub, Exp, ReduceSum, Div
}

// ========== 写回 GM ==========
AscendC::DataCopyPad(yGm[offset], yOut,
    {static_cast<uint16_t>(rows), static_cast<uint32_t>(rLength * sizeof(T)), 0, 0, 0});
```

**关键对照表**：

| 参数位置 | 用 rLength | 用 rLengthAlign |
|---------|-----------|-----------------|
| DataCopyPad blockLen | ✓ | ✗ |
| Reduce API count | ✓ | ✗ |
| Sub/Exp/Div count | ✓ | ✗ |
| UB rowOffset | ✗ | ✓ |
| Buffer 大小 | ✗ | ✓ |

**UB 数据布局示意**：

```
GM（连续存储）:  [row0: 13元素][row1: 13元素][row2: 13元素]...
                         ↓ DataCopyPad blockCount 模式
UB（对齐存储）:  [row0: 13+3=16][row1: 13+3=16][row2: 13+3=16]...
                         ↑
                  每行 padding 到 8 元素对齐
```

## stride 参数详解

**stride 参数单位取决于操作数位置**：

| 操作数位置 | stride 单位 | 说明 |
|-----------|------------|------|
| GlobalTensor (GM) | **字节** | 相邻数据块的字节间隔 |
| LocalTensor (UB) | **dataBlock (32字节)** | 相邻数据块的 32字节块间隔 |

**stride 含义**：相邻数据块之间的间隔（前一块尾部到后一块头部的距离）

### UB → GM 多行搬运（CopyOut）

```cpp
// UB 中每行: [cols 有效数据][padElements padding]
// 相邻行间隔 = paddedColsT - cols 个元素
copyParams.blockCount = rowsThisTile;
copyParams.blockLen = cols * sizeof(T);
copyParams.srcStride = (paddedColsT - cols) * sizeof(T) / 32;  // UB stride 单位: 32字节
copyParams.dstStride = 0;  // GM stride 单位: 字节

AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 常见错误

```cpp
// ❌ 错误：srcStride 理解为行长度
copyParams.srcStride = paddedColsT * sizeof(T) / 32;  // 这会导致输出错位

// ✅ 正确：srcStride 是间隔
copyParams.srcStride = (paddedColsT - cols) * sizeof(T) / 32;
```

## 常见错误与调试

### 错误1：CopyIn/CopyOut 非对齐数据用 DataCopy

```cpp
// ❌ 错误
AscendC::DataCopy(xLocal, xGm, 4);  // cols=4 (16 bytes)，数据错误

// ✅ 正确
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
```

### 错误2：CopyIn 用 DataCopyPad，CopyOut 用 DataCopy

```cpp
// ❌ 错误：CopyIn 和 CopyOut 都需要处理非对齐
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
AscendC::DataCopy(yGm, yLocal, 4);  // 输出错误

// ✅ 正确：两边都用 DataCopyPad
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

### 调试步骤

遇到数据错误时：

1. **分别验证 CopyIn 和 CopyOut**
   - 用 "CopyIn → CopyOut" 测试搬运是否正确
2. **检查数据量是否 32 字节对齐**
3. **非对齐场景：CopyIn 和 CopyOut 都用 DataCopyPad**

### 实战案例：SoftmaxV5

**问题**：FP32 cols=4,5,6,7 时结果错误，cols=8 正常

**根因**：
1. CopyIn 用 DataCopyPad 但 isPad=false（填充随机值）
2. CopyOut 用 DataCopy 处理非对齐输出

**解决**：CopyIn 和 CopyOut 都用 DataCopyPad

---

# UB 缓冲区管理指南

TBuf/TQue 选择、Double Buffer 流水线并行、批量搬运模式。

## TBuf vs TQue 选择

| 场景 | 推荐类型 | 说明 |
|-----|---------|------|
| MTE2/MTE3 搬运缓冲区 | `TQue<VECIN/VECOUT>` | 需要与 Vector 并行，需要 EnQue/DeQue |
| 纯 Vector 计算缓冲区 | `TBuf<VECCALC>` | 不涉及 MTE 搬运，用 `Get<T>()` 获取 |
| Double Buffer | `TQue` + `InitBuffer(que, 2, size)` | 在 InitBuffer 中设置 num=2 开启 |

## TQue 详解

### 模板参数

```cpp
template <TPosition pos, int32_t depth, auto mask = 0> class TQue;
```

| 参数 | 说明 |
|------|------|
| `pos` | 队列逻辑位置：`VECIN`, `VECOUT`, `A1`, `A2`, `B1`, `B2`, `CO1`, `CO2` |
| `depth` | 队列深度，表示可连续 EnQue/DeQue 的次数 |
| `mask` | 数据格式转换（ND↔NZ）或编译期优化参数 |

### depth 参数关键说明

| depth 值 | 适用场景 | 说明 |
|---------|---------|------|
| `depth=1` | **默认推荐**，非 Tensor 原地操作 | 编译器有特殊优化，性能更好 |
| `depth=0` | **Tensor 原地操作** | 需要设置 |
| `depth=2` | 连续 2 次 EnQue 场景 | 与 InitBuffer 的 num 参数独立 |

**注意**：`depth` 与 Double Buffer 无关。Double Buffer 由 `InitBuffer` 的 `num` 参数控制。

```cpp
// ✅ 非连续入队（普通场景）：depth=1 即可
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;
pipe->InitBuffer(que, 1, size);
auto tensor = que.AllocTensor<T>();
que.EnQue(tensor);
tensor = que.DeQue<T>();
que.FreeTensor(tensor);
```

### Double Buffer 配置

**Double Buffer 是在 `InitBuffer` 的 `num` 参数中设置，与模板参数 `depth` 无关。**

| InitBuffer 参数 | 作用 | 说明 |
|----------------|------|------|
| `InitBuffer(que, num, size)` | `num` 控制 Double Buffer | `num=1`=单 Buffer，`num=2`=开启 Double Buffer |
| 模板参数 `depth` | 队列深度 | 表示可连续 EnQue 的次数 |

```cpp
// ✅ 开启 Double Buffer：在 InitBuffer 中设置 num=2
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;  // 模板 depth=1 即可
pipe->InitBuffer(que, 2, size);  // num=2 开启 Double Buffer

// ✅ 关闭 Double Buffer
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;
pipe->InitBuffer(que, 1, size);  // num=1 单 Buffer
```

### TQue Buffer 数量限制

| 产品系列 | eventID 数量 | 最大 TQue 数量 |
|---------|-------------|---------------|
| Atlas 训练系列 | 4 | 4 |
| Atlas 推理系列 AI Core | 8 | 8 |
| Atlas 推理系列 Vector Core | 8 | 8 |
| Atlas A2/A3 系列 | 8 | 8 |

**注意**：
- 不开启 Double Buffer（num=1）：最多可申请 8 个 TQue
- 开启 Double Buffer（num=2）：每个 TQue 占用 2 个 buffer，最多只能申请 4 个 TQue

```cpp
// 开启 Double Buffer 时，最多只能申请 4 个 TQue
pipe->InitBuffer(que0, 2, size);  // ✅
pipe->InitBuffer(que1, 2, size);  // ✅
pipe->InitBuffer(que2, 2, size);  // ✅
pipe->InitBuffer(que3, 2, size);  // ✅
pipe->InitBuffer(que4, 2, size);  // ❌ 超过限制
```

### TQue 正确用法

```cpp
// TQue：需要队列管理（MTE 搬运相关）
// 模板 depth=1 即可，Double Buffer 在 InitBuffer 的 num 参数中设置
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
pipe->InitBuffer(inQueueX, 2, bufferSize);  // num=2 开启 Double Buffer

AscendC::LocalTensor<half> x = inQueueX.AllocTensor<half>();
AscendC::DataCopyPad(x, xGm, {1, size * sizeof(half), 0, 0}, {false, 0, 0, 0});
inQueueX.EnQue(x);
// ...
AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
inQueueX.FreeTensor(xLocal);
```

## TBuf 详解

### 特性

| 特性 | 说明 |
|------|------|
| 内存用途 | 只能参与计算，无法执行 EnQue/DeQue |
| 内存分配 | 每次 InitBuffer 只分配一块内存 |
| Tensor 释放 | 无需手动释放 |

```cpp
// TBuf：纯计算缓冲区
AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
pipe->InitBuffer(workBuf, bufferSize);

// ✅ 使用 Get<T>() 获取 Tensor，无需释放
AscendC::LocalTensor<float> work = workBuf.Get<float>();
// ... 计算逻辑 ...
// 无需 FreeTensor
```

## Double Buffer 流水线并行

### 核心认知

**Double Buffer 不是"用2块内存计算"，而是"用2块内存做搬入/搬出，使 MTE2/MTE3 与 Vector 计算并行"。**

本质：**内存搬运与计算并行，掩盖搬运延迟**。

### 硬件原理

- **MTE2**：搬运工，GM → UB
- **Vector**：加工员，计算
- **MTE3**：搬运工,UB → GM

### 时间线对比

**无 Double Buffer（串行）**：
```
Row 0: [MTE2][Vector][MTE3]
Row 1:                      [MTE2][Vector][MTE3]
```

**有 Double Buffer（并行）**：
```
Row 0: [MTE2-B0][Vector-B0][MTE3-B0]
Row 1:          [MTE2-B1][Vector-B1][MTE3-B1]
                  ↑ MTE2与Vector并行！
```

### 实现原则

| Buffer 类型 | InitBuffer num | 说明 |
|------------|----------------|------|
| `TQue<VECIN>` (MTE2 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TQue<VECOUT>` (MTE3 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TBuf<VECCALC>` (纯计算) | - | TBuf 不涉及 MTE 搬运 |

### 正确用法

```cpp
// 1. Init: num=2 开启 Double Buffer
pipe->InitBuffer(inQueueX,  2, tileSize * sizeof(T));
pipe->InitBuffer(outQueueY, 2, tileSize * sizeof(T));
pipe->InitBuffer(workBuf, workSize * sizeof(T));

// 2. Process: 单循环结构，TQue 自动轮转
for (int i = 0; i < totalTiles; i++) {
    CopyIn(i);   // MTE2 异步搬运
    Compute(i);  // Vector 计算
    CopyOut(i);  // MTE3 异步搬出
}

// 3. CopyIn
void CopyIn(int i) {
    LocalTensor<T> x = inQueueX.AllocTensor<T>();
    DataCopyPad(x, xGm[i * tileSize], {1, (uint32_t)(tileSize * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    inQueueX.EnQue(x);
}

// 4. Compute
void Compute(int i) {
    LocalTensor<T> x = inQueueX.DeQue<T>();
    LocalTensor<T> y = outQueueY.AllocTensor<T>();
    Add(y, x, constTensor, tileSize);
    outQueueY.EnQue(y);
    inQueueX.FreeTensor(x);
}

// 5. CopyOut
void CopyOut(int i) {
    LocalTensor<T> y = outQueueY.DeQue<T>();
    DataCopyPad(yGm[i * tileSize], y, {1, (uint32_t)(tileSize * sizeof(T)), 0, 0});
    outQueueY.FreeTensor(y);
}
```

### 为什么能并行？

| 操作 | 特性 |
|------|------|
| `DataCopy` | 异步 DMA，立即返回 |
| `EnQue` | 非阻塞，标记就绪 |
| `DeQue` | 阻塞，等待就绪 |

### 常见误区

| 误区 | 正确理解 |
|------|---------|
| 需要手动拆成 Ping/Pong 两套代码 | 单循环 + `InitBuffer(que, 2, size)` 自动管理 |
| depth 模板参数控制 Double Buffer | Double Buffer 由 `InitBuffer` 的 `num` 参数控制 |
| depth 越大越好 | 模板 depth 通常设为 1，性价比最高 |
| 所有 buffer 都要 num=2 | 只有涉及 MTE 搬运的才需要 Double Buffer |

## 批量搬运 + 逐行计算模式

### 适用场景

处理多行数据时，批量搬运减少 MTE2/MTE3 调用次数，充分利用带宽。

### 模式结构

```
CopyInBatch(N行) → 逐行计算(N次) → CopyOutBatch(N行)
```

### 代码模板

```cpp
__aicore__ inline void ProcessBatch()
{
    uint32_t totalRowsToProcess = endRow - startRow;
    if (totalRowsToProcess == 0) return;

    for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
        uint32_t startLocalRow = tile * tileRows;

        // 边界检查：防止 uint32_t 下溢
        if (startLocalRow >= totalRowsToProcess) break;

        uint32_t remaining = totalRowsToProcess - startLocalRow;
        uint32_t rowsThisTile = (remaining < tileRows) ? remaining : tileRows;

        CopyInBatch(startLocalRow, rowsThisTile);
        ComputeBatch(rowsThisTile);
        CopyOutBatch(startLocalRow, rowsThisTile);
    }
}
```

### Host 侧 Tiling 计算

```cpp
// A2/A3 UB = 192KB
constexpr uint64_t UB_SIZE = 192 * 1024;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;  // DataCopyPad blockCount 限制

// bytesPerTileRow: double buffer (in*2 + out*2)
uint32_t bytesPerTileRow = paddedColsT * typeSizeBytes * 4;

// tileRows
uint32_t tileRows = (UB_SIZE - overheadBytes) / bytesPerTileRow;
tileRows = std::max(1u, std::min(tileRows, MAX_BLOCK_COUNT));
```

### 注意事项

1. **tileRows 限制**：DataCopyPad 的 `blockCount` 最大 4095
2. **尾核处理**：`startLocalRow >= totalRowsToProcess` 时提前退出
3. **stride 计算**：UB 侧 stride 单位是 32 字节块，GM 侧是字节

---

# 精度转换与混合精度指南

Cast API 使用规范和混合精度计算模式。

## Cast RoundMode 选择

### 选择规则

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| **half → float** | `CAST_NONE` | 低精度→高精度，无精度损失 |
| **float → half** | `CAST_ROUND` | 高精度→低精度，有精度损失 |
| half → int32_t | `CAST_ROUND` / `CAST_CEIL` | 量化场景，根据需求选择 |
| int32_t → float | `CAST_NONE` | 整数→浮点，无精度损失 |

### 正确用法

```cpp
// ✅ half → float：低精度到高精度
AscendC::LocalTensor<float> xFloat = workBuf.Get<float>();
AscendC::Cast<float, half>(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, count);

// ✅ float → half：高精度到低精度
AscendC::LocalTensor<half> yHalf = outQueue.AllocTensor<half>();
AscendC::Cast<half, float>(yHalf, xFloat, AscendC::RoundMode::CAST_ROUND, count);
```

## 混合精度计算模式（FP16 输入）

### 适用场景

当输入输出为 FP16，但需要 FP32 精度进行中间计算时（如 Softmax、LayerNorm）。

### 计算流程

```
half 输入 → Cast(FP32) → 中间计算(FP32) → Cast(half) → half 输出
```

### 为什么需要 FP32 中间计算？

1. **ReduceMax/Exp/ReduceSum** 在 FP32 上精度更稳定
2. **避免 FP16 数值溢出**：Exp 结果可能超出 FP16 表示范围
3. **累积误差控制**：多次运算的累积误差在 FP32 下更小

---

# 流水线同步机制指南

MTE 与 Vector 同步的核心机制。

## 核心问题

**DataCopy/DataCopyPad 是异步 DMA 操作，直接在搬运后的数据上做 Vector 计算可能读到未完成的数据！**

### 硬件架构

```
GM → MTE2 (异步) → UB → Vector (同步) → MTE3 (异步) → GM
```

### 问题场景

```cpp
// ❌ 错误：DataCopyPad 后直接使用数据
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);  // 可能读到未完成搬运的数据！
```

**现象**：输出数据随机、错误

## 解决方案

### 方案一：EnQue/DeQue 队列同步（推荐）

**原理**：TQue 的 EnQue/DeQue 机制自动提供硬件同步点。

```cpp
// ✅ 正确：使用 EnQue/DeQue 同步
// Step 1: CopyIn - MTE2 搬运
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
inQueueX.EnQue(xLocal);                    // 标记"就绪"

// Step 2: Compute - Vector 计算
AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // 阻塞等待 MTE2 完成
AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
AscendC::Adds<float>(yLocal, xIn, 1.0f, count);
outQueueY.EnQue(yLocal);
inQueueX.FreeTensor(xIn);

// Step 3: CopyOut - MTE3 搬运
AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // 阻塞等待 Vector 完成
AscendC::DataCopyPad(yGm[gmOffset], yOut, copyOutParams);
outQueueY.FreeTensor(yOut);
```

**关键点**：
- `EnQue(xLocal)` 标记 buffer 数据就绪
- `DeQue<float>()` 阻塞等待数据就绪
- DeQue 返回后，数据一定已经搬运完成

### 方案二：PipeBarrier 手动同步

```cpp
// ✅ 可用：使用 PipeBarrier 同步
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
AscendC::PipeBarrier<PIPE_ALL>();          // 等待 MTE2 完成

AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);

AscendC::DataCopyPad(yGm[gmOffset], yLocal, copyOutParams);
AscendC::PipeBarrier<PIPE_ALL>();          // 等待 MTE3 完成
```

**缺点**：性能开销大（全流水线停顿），不推荐用于高性能场景

## 两种方案对比

| 特性 | EnQue/DeQue | PipeBarrier |
|-----|-------------|-------------|
| 同步粒度 | buffer 级别 | 全流水线 |
| 性能 | 高（支持并行） | 低（串行等待） |
| 代码复杂度 | 需要队列管理 | 简单直接 |
| 推荐程度 | ⭐⭐⭐⭐⭐ | ⭐⭐（仅调试用） |

### EnQue/DeQue 的双重作用

1. **队列管理**：Double Buffer 场景下管理多 buffer 轮转
2. **硬件同步**：提供 MTE ↔ Vector 之间的同步点

```cpp
// EnQue/DeQue 不仅仅是"队列"，更重要的是同步机制
inQueueX.EnQue(xLocal);    // 1. 标记数据就绪  2. 通知硬件可以等待
xLocal = inQueueX.DeQue(); // 1. 阻塞等待就绪  2. 获取可用 buffer
```

## 完整流水线模板

```cpp
__aicore__ inline void ProcessTile(uint32_t tileIdx)
{
    // ========== CopyIn 阶段 ==========
    // MTE2: GM → UB（异步）
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopyPad(xLocal, xGm[tileIdx * tileSize], copyParams, padParams);
    inQueueX.EnQue(xLocal);              // 同步点：标记就绪

    // ========== Compute 阶段 ==========
    // Vector: UB 计算（同步，需等待 MTE2）
    AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // 同步点：等待 MTE2
    AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    AscendC::Adds<float>(yLocal, xIn, 1.0f, tileSize);
    outQueueY.EnQue(yLocal);             // 同步点：标记就绪
    inQueueX.FreeTensor(xIn);

    // ========== CopyOut 阶段 ==========
    // MTE3: UB → GM（异步，需等待 Vector）
    AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // 同步点：等待 Vector
    AscendC::DataCopyPad(yGm[tileIdx * tileSize], yOut, copyParams);
    outQueueY.FreeTensor(yOut);
}
```

### 流水线时序图

```
时间 →

Tile 0:  [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                      ↑ DeQue等待    ↑ DeQue等待
Tile 1:          [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                  ↑ 并行！    ↑ DeQue等待    ↑ DeQue等待

关键：DeQue 阻塞等待上一个阶段的异步操作完成
```

## 调试技巧

### 检查缺少 EnQue/DeQue

```cpp
// ❌ 错误：AllocTensor 后直接用
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
Compute(x);  // 错！可能读到未完成搬运的数据

// ✅ 正确：DeQue 后再计算
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();  // 等待搬运完成
Compute(xIn);
```

### 临时加 PipeBarrier 调试

```cpp
DataCopy(x, gm, size);
PipeBarrier<PIPE_ALL>();  // 临时加，如果结果正确说明是同步问题
Compute(x);
```

**如果 PipeBarrier 能解决问题，说明是同步问题** → 修复方案：改为 EnQue/DeQue 机制

### 常见误区

| 误区 | 正确理解 |
|-----|---------|
| AllocTensor 后数据就可用 | AllocTensor 只分配内存，不等待搬运 |
| DataCopy 是同步的 | DataCopy 是异步 DMA，立即返回 |
| 不用 EnQue/DeQue 也能正常工作 | 必须用 EnQue/DeQue 或 PipeBarrier 同步 |
| PipeBarrier 性能好 | PipeBarrier 是全流水线停顿，性能差 |

---

# Vector API repeatTime 参数限制

> **核心问题**：repeatTime 为 uint8_t 时最大值 255，超过会溢出导致计算错误

## 核心约束

**使用 Vector API 前，必须确认 `repeatTime` 参数的范围限制！**

当 `repeatTime` 为 `uint8_t` 类型时，最大值 **255**。

## 如何确认参数范围

### 1. 查看函数原型中的参数类型

```cpp
// Sub API 函数原型
template <typename T, bool isSetMask = true>
__aicore__ inline void Sub(
    const LocalTensor<T>& dst,
    const LocalTensor<T>& src0,
    const LocalTensor<T>& src1,
    uint64_t mask,
    const uint8_t repeatTime,  // ← uint8_t 类型
    const BinaryRepeatParams& repeatParams
);
```

### 2. 根据数据类型确定范围

| 数据类型 | 取值范围 | 含义 |
|---------|---------|------|
| `uint8_t` | 0 ~ 255 | 最多 255 次迭代 |
| `uint16_t` | 0 ~ 65535 | 最多 65535 次迭代 |
| `int32_t` | - | 无此限制 |

## 问题场景与解决方案

### 问题场景

```cpp
// 问题：R=256，rowCount=256 > 255
uint32_t rowCount = 256;
AscendC::Sub<float>(
    dst[col],
    src0[col],
    src1[col],
    curMask,
    rowCount,  // uint32_t 传入 uint8_t 参数，256 溢出为 0！
    {1, 1, 1, repStride, repStride, 0}
);
```

**结果**：`rowCount=256` 被截断为 `0`，Sub 不执行任何计算，输出数据错误。

### 方案一：Host 侧限制 R_max（推荐）

```cpp
// Host 侧 R_max 计算
constexpr uint32_t MAX_REPEAT_TIMES = 255;
uint32_t R_max = (UB_SIZE - overheadBytes) / bytesPerRow;
R_max = std::min(R_max, MAX_REPEAT_TIMES);  // 确保 R_max <= 255
```

### 方案二：Kernel 侧分批处理

```cpp
void SubWithBroadcast(
    AscendC::LocalTensor<float>& dst,
    AscendC::LocalTensor<float>& src0,
    AscendC::LocalTensor<float>& src1,
    uint32_t a0Count,
    uint32_t alignedCols,
    uint32_t rowCount)
{
    constexpr uint32_t MAX_REPEAT = 255;
    uint32_t repStride = alignedCols / BLOCK_ELEMENTS;  // BLOCK_ELEMENTS = 8

    for (uint32_t col = 0; col < a0Count; col += MASK_FP32) {
        uint32_t curMask = std::min(a0Count - col, MASK_FP32);

        // 分批处理
        uint32_t processedRows = 0;
        while (processedRows < rowCount) {
            uint32_t batchRepeat = std::min(rowCount - processedRows, MAX_REPEAT);
            uint32_t rowOffset = processedRows * alignedCols;

            AscendC::Sub<float>(
                dst[rowOffset + col],
                src0[rowOffset + col],
                src1[col],
                curMask,
                batchRepeat,
                {1, 1, 1, static_cast<uint8_t>(repStride), static_cast<uint8_t>(repStride), 0}
            );

            processedRows += batchRepeat;
        }
    }
}
```

## 受影响的 API

| API 类别 | API 名称 | 参数限制 |
|---------|---------|---------|
| 二元运算 | Sub, Add, Mul, Div, Max, Min | `repeatTime` ≤ 255 |
| 一元运算 | Exp, Log, Sqrt, Abs, Neg | `repeatTime` ≤ 255 |
| 标量运算 | Muls, Adds, Divs | `repeatTime` ≤ 255 |
| 其他 | And, Or, Xor, Not | `repeatTime` ≤ 255 |

## 最佳实践

| 阶段 | 检查项 |
|-----|-------|
| **API 使用前** | 查看文档确认 `repeatTime` 数据类型和范围 |
| **Host Tiling** | 确保 `R_chunk_size` / `tileRows` 不超过限制 |
| **Kernel 实现** | 如需超过限制，实现分批处理逻辑 |
| **调试** | 若 R=256 时出错，优先检查 repeatTime 溢出 |

---

# API 使用限制与替代方案

> **重要**：使用任何 API 前必读，避免编译错误和运行时问题

## 1. 编译期限制

### 1.1 禁止使用 std:: 计算函数

**原因**：Kernel 侧不支持 C++ 标准库，必须使用 Ascend C 提供的专用 API

**触发场景**：所有数学计算、比较操作

**禁止列表**：

| std:: 函数 | ❌ 错误用法 | ✅ Ascend C 替代 | 说明 |
|-----------|----------|----------------|------|
| `std::abs` | `std::abs(x)` | `AscendC::Abs(dst, src, count)` | 绝对值 |
| `std::min/max` | `std::min(a, b)` | `(a < b) ? a : b` 或 `AscendC::Min/Max` | 最小/最大值 |
| `std::sqrt` | `std::sqrt(x)` | `AscendC::Sqrt(dst, src, count)` | 平方根 |
| `std::pow` | `std::pow(x, y)` | `AscendC::Power(dst, src, count)` | 幂运算 |
| `std::exp` | `std::exp(x)` | `AscendC::Exp(dst, src, count)` | 指数 |
| `std::log/log2/log10` | `std::log(x)` | `AscendC::Log/Log2/Log10(dst, src, count)` | 对数 |
| `std::sin/cos/tan` | `std::sin(x)` | `AscendC::Sin/Cos/Tan(dst, src, count)` | 三角函数 |
| `std::floor/ceil/round` | `std::floor(x)` | `AscendC::Floor/Ceil/Round(dst, src, count)` | 取整 |
| `std::isnan/isinf` | `std::isnan(x)` | 手动检查 | 特殊值判断 |

**错误示例**：
```cpp
#include <algorithm>
#include <cmath>

uint32_t result = std::min(a, b);  // ❌ 编译错误
float val = std::sqrt(x);          // ❌ 编译错误
float val = std::exp(x);           // ❌ 编译错误
```

**正确替代**：
```cpp
// min/max：使用三元操作符
uint32_t result = (a < b) ? a : b;  // ✅ min
uint32_t result = (a > b) ? a : b;  // ✅ max

// 或使用 Ascend C API（批量操作）
AscendC::LocalTensor<T> minLocal = minBuf.Get<T>();
AscendC::LocalTensor<T> srcLocal = srcBuf.Get<T>();
AscendC::Min<T>(minLocal, srcLocal, src2Local, count);  // ✅ 批量最小值

// sqrt/exp/log 等：使用 Ascend C API
AscendC::LocalTensor<T> dstLocal = dstBuf.Get<T>();
AscendC::LocalTensor<T> srcLocal = srcBuf.Get<T>();
AscendC::Sqrt<T>(dstLocal, srcLocal, count);  // ✅ 平方根
AscendC::Exp<T>(dstLocal, srcLocal, count);   // ✅ 指数
AscendC::Log<T>(dstLocal, srcLocal, count);   // ✅ 对数
```

**⚠️ 重要**：所有数学计算都必须使用 Ascend C API，不能混用 std:: 函数！

### 1.2 禁止动态内存分配

**原因**：AI Core 无动态内存管理能力

**触发场景**：创建数组、缓冲区等

**错误示例**：
```cpp
std::vector<int> vec;       // ❌ 动态分配
int* ptr = new int[10];     // ❌ 动态分配
int* arr = malloc(100);     // ❌ 动态分配
```

**正确替代**：使用静态分配
```cpp
int arr[10];                          // ✅ 栈分配（Host 侧）
constexpr uint32_t SIZE = 1024;       // ✅ 编译期常量
pipe.InitBuffer(inQueue, 2, SIZE);    // ✅ UB 静态分配（Kernel 侧）
```

### 1.3 Host/Kernel 头文件隔离

**规则**：
- **Host 侧**（`.cpp`）：禁止包含 `kernel_operator.h`
- **Kernel 侧**（`.asc/.h`）：可包含 `kernel_operator.h`

**错误示例**：
```cpp
// host/tiling.cpp
#include "kernel_operator.h"  // ❌ Host 侧禁止
```

**正确用法**：
```cpp
// host/tiling.cpp
#include "tiling.h"  // ✅ 仅必要头文件
#include <cstring>

// kernel/operator.h
#include "kernel_operator.h"  // ✅ Kernel 侧允许
```

## 2. API 使用限制索引

以下限制在各专题文档中详细说明：

| 限制类型 | 详细文档 | 核心要点 |
|---------|---------|---------|
| **GM 数据搬运** | DataCopy 章节 | 禁用 SetValue/GetValue，强制 DataCopyPad |
| **Reduce API** | Reduce 章节 | dst ≠ tmpBuffer，禁用低阶 API |
| **Compare 256字节对齐** | 见下文 2.1 | count 需 256B 对齐，padding 策略 |
| **repeatTime 限制** | Repeat-limits 章节 | uint8_t 最大 255，需分批处理 |
| **流水线同步** | Pipeline 章节 | MTE/Vector 必须用 EnQue/DeQue 同步 |

### 2.1 Compare API 256字节对齐约束

**约束**：`count` 个元素所占空间必须 **256 字节对齐**

**处理方案**：Padding 策略

```cpp
// 1. 计算对齐大小（float 类型：64 的倍数）
constexpr uint32_t A0 = 32;
constexpr uint32_t A0_ALIGN = (A0 + 63) / 64 * 64;  // = 64

// 2. UB Buffer 使用对齐大小
pipe.InitBuffer(inQueue, 1, R * A0_ALIGN * sizeof(float));

// 3. CopyIn 时填充极值
Duplicate(xLocal, -FLT_MAX, R * A0_ALIGN);  // ArgMax 用极小值
// 再拷贝实际数据到前 A0 个位置

// 4. API 调用使用对齐大小
Compare(cmpLocal, srcLocal, maxLocal, CMPMODE::GT, A0_ALIGN);

// 5. CopyOut 只输出有效数据
DataCopy(dstGm, yLocal, A0);  // 只输出 A0 个
```

**极值选择**：
- ArgMax / 找最大值：`-FLT_MAX` 或 `-INFINITY`
- ArgMin / 找最小值：`FLT_MAX` 或 `INFINITY`

## 3. 类型与常量规范

### 3.1 编译期常量

**规则**：Buffer 大小、循环次数等使用 `constexpr`

```cpp
// ✅ 正确：编译期常量
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t BLOCK_SIZE = 32;

// ❌ 不推荐：运行期常量
const uint32_t buffer_num = 2;  // 可能影响性能
```

### 3.2 类型转换

**规则**：显式类型转换，避免隐式精度损失

```cpp
// ✅ 正确：显式转换
T sumVal = scalarLocal.GetValue(0);
T invSumVal = (T)1.0 / sumVal;  // 显式转换为 T
Muls<T>(dst, src, invSumVal, count);

// ❌ 错误：隐式转换
float val = 1.0 / sumVal;  // 若 T 是 half，精度损失
```

## 4. 快速诊断清单

遇到编译错误时，检查：

- [ ] 是否使用了 **任何 std:: 计算函数**（min/max/abs/sqrt/exp/log等）→ 改用 Ascend C API 或基础操作
- [ ] 是否使用了动态内存（`std::vector`, `new`）→ 改用静态分配
- [ ] Host 侧是否包含了 `kernel_operator.h` → 移除该包含
- [ ] Reduce API 的 dst 和 tmp 是否是同一 buffer → 使用不同 buffer
- [ ] 是否使用了 WholeReduce 等低阶 API → 改用高阶 Reduce API
- [ ] 是否使用了 `const` 而非 `constexpr` → 改用 `constexpr`
- [ ] 是否使用了不存在的类型（如 `TensorShape`）→ 查阅正确 API
- [ ] Compare API 的 count 是否满足 256 字节对齐 → 使用 padding 策略

---

# Host 侧 Runtime API 使用规范

> **适用范围**：Kernel 直调模式下的 Host 侧代码（`.asc` 中的 `main()` 函数）

## 1. 设备初始化 API 调用顺序 ⚠️ **强制**

### 1.1 核数获取 API 选择 ⚠️ **关键**

**根据算子类型选择正确的核数获取 API**：

| 算子类型 | 使用的 API | 说明 |
|---------|-----------|------|
| **纯向量计算**（Add/Mul/Div/Reduce等） | `ACL_DEV_ATTR_VECTOR_CORE_NUM` | 使用 Vector Core 数量 |
| **矩阵计算**（MatMul/Conv等） | `ACL_DEV_ATTR_CUBE_CORE_NUM` | 使用 Cube Core 数量 |
| **混合计算** | `ACL_DEV_ATTR_AICORE_CORE_NUM` | 使用 AI Core 数量 |

**910B3 芯片核数参考**：
- AI Core: 20
- Cube Core: 20
- Vector Core: 40（每个 AI Core 有 2 个 Vector Core）

### 1.2 aclrtGetDeviceInfo 调用要求

**规则**：`aclrtGetDeviceInfo` **必须**在 `aclrtSetDevice` 之后调用

**原因**：获取设备资源前必须先设置设备上下文

**正确示例**（纯向量算子）：
```cpp
int32_t main() {
    // 1. 初始化 ACL
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSetDevice failed, ret=%d\n", ret);
        return ret;
    }

    // 2. 获取设备核数（必须在 aclrtSetDevice 之后）
    int64_t availableCoreNum = 8;  // 默认值
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    if (ret != ACL_SUCCESS) {
        printf("aclrtGetDeviceInfo failed, ret=%d\n", ret);
        aclrtResetDevice(deviceId);
        return ret;
    }

    // 3. 计算使用核数
    uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

    // 4. 后续处理...
}
```

**矩阵算子示例**：
```cpp
// 矩阵计算算子使用 Cube Core 数量
int64_t availableCoreNum = 8;
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &availableCoreNum);
```

**错误示例**：
```cpp
int32_t main() {
    // ❌ 错误：未调用 aclrtSetDevice 就调用 aclrtGetDeviceInfo
    int64_t availableCoreNum = 8;
    aclError ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    // 可能返回错误或获取到错误值
}
```

## 2. 常见错误

| 错误类型 | 错误示例 | 后果 | 正确做法 |
|---------|---------|------|---------|
| **调用顺序错误** | 未调用 `aclrtSetDevice` 就调用 `aclrtGetDeviceInfo` | 获取核数失败或返回错误值 | 先 `aclrtSetDevice`，再获取资源 |
| **写死核数** | `uint32_t numBlocks = 8;` | 不同设备性能不匹配 | 使用 `aclrtGetDeviceInfo` 动态获取 |
| **API 选择错误** | 纯向量算子用 `ACL_DEV_ATTR_AICORE_CORE_NUM` | 未充分利用 Vector Core | 根据算子类型选择正确的 API |

## 3. 完整 Host 侧初始化流程

```cpp
int32_t main() {
    // Step 1: 初始化 ACL
    aclInit(nullptr);

    // Step 2: 设置设备
    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    CHECK_ACL(ret);

    // Step 3: 获取设备核数（根据算子类型选择）
    int64_t availableCoreNum = 8;
    // 纯向量算子
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    // 或矩阵算子：ACL_DEV_ATTR_CUBE_CORE_NUM
    // 或混合算子：ACL_DEV_ATTR_AICORE_CORE_NUM
    CHECK_ACL(ret);

    // Step 4: 分配 GM 内存
    size_t gmSize = ...;
    void* gmPtr = nullptr;
    ret = aclrtMalloc(&gmPtr, gmSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_ACL(ret);

    // Step 5: 计算 Tiling 参数
    MyTilingData tiling;
    uint32_t numBlocks = (uint32_t)availableCoreNum;
    computeTiling(tiling, totalRows, numBlocks);

    // Step 6: 启动 Kernel
    KernelCall(..., (uint8_t*)&tiling);

    // Step 7: 清理资源
    aclrtFree(gmPtr);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
