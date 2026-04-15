# Qwen3 Coder 30B A3B Instruct 全量算子基准执行设计

## 目标
使用 MultiKernelBench 现有测试框架，对模型 `qwen3-coder-30b-a3b-instruct` 执行全量基准测试，覆盖数据集中映射到 `reference/` 的全部算子，即使用 `--categories all`。

## 执行范围
- 模型名：`qwen3-coder-30b-a3b-instruct`
- Prompt 策略：`add_shot`
- 运行次数：`1`
- 类别范围：`all`
- 平台要求：阿里云平台路由到对应模型服务

## 执行流程
### 1. 生成阶段
运行：
```bash
python generate_and_write.py --model qwen3-coder-30b-a3b-instruct --strategy add_shot --categories all
```

预期行为：
- 遍历数据集中的全部算子。
- 为每个算子生成 prompt 并请求模型输出。
- 将生成结果写入：
  `output/ascendc/add_shot/{temperature}-{top_p}/qwen3-coder-30b-a3b-instruct/run0/`

### 2. 评测阶段
运行：
```bash
python evaluation.py --model qwen3-coder-30b-a3b-instruct --strategy add_shot --categories all
```

预期行为：
- 读取生成阶段保存的每个算子输出。
- 调用 `eval_single_runner.py` 完成单算子编译、正确性校验与性能测试。
- 将最终结果写入：
  `output/ascendc/add_shot/{temperature}-{top_p}/qwen3-coder-30b-a3b-instruct/run0/result.json`

## 输出物
本次执行完成后需要产出：
- 生成输出目录
- 全量评测结果 JSON
- 结果汇总，包括：
  - 总算子数
  - 成功生成数
  - 编译成功数
  - 正确性通过数
  - 有性能数据的算子数
  - 典型失败原因

## 处理原则
- 不修改仓库代码，只使用现有框架执行。
- 若模型名在阿里云平台不可用，则停止执行并报告报错。
- 若出现平台限流、网络异常或权限问题，则保留日志并报告。
- 若出现 Ascend/NPU 环境缺失，则停止并报告缺失项。
- 若仅个别算子失败，则按现有评测框架继续执行并在最终汇总中说明。

## 完成标准
满足以下条件之一即可认为本次任务完成：
1. generation 与 evaluation 均成功跑完 `--categories all`；或
2. generation 成功完成且 evaluation 输出完整 `result.json`，即使部分算子失败。

最终需要向用户返回：
- 实际执行命令
- 结果文件路径
- 关键统计摘要
- 主要失败原因或阻塞项
