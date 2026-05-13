# 使用方式
评测分为两步：算子生成 ——> 生成算子上板运行测试

## 执行算子生成

`batch_scripts/batch_01_wikiV0.sh` 是示例生成脚本。默认跑 `selected_ops.yaml` 的 **40 个 LLM 算子**（完整评测）；
带 `--smoke` 跑 `selected_ops_smoke.yaml` 的 **10 个算子**（快速验证）。运行前主要注意改自己的实验任务名 `model-name`。

```bash
bash batch_scripts/batch_01_wikiV0.sh            # 40 个 LLM 算子 (完整)
bash batch_scripts/batch_01_wikiV0.sh --smoke    # 10 个 LLM 算子 (快速验证)
```

> 注：`--strategy add_shot`（脚本默认值）已切换到 anti-hack 加固版模板，
> 自动嵌入算子 schema + 反例段 + ANTI-HACK RULES，引导模型按 schema 实现 custom op 而非
> 复制 reference 的 PyTorch 实现。原 add_shot 模板已下线，调用方式无需改动。

**注意事项**
1. 评测中可能出现生成超时的算子，这种要重跑，不算生成失败
2. 评测中可能出现`[SKIP] {op}: output looks like rate-limit/error`，这种可能是API消耗超限，也要后续重跑，不算生成失败
3. 评测中可能出现`[SKIP] {op}: could not locate all six required variables`，这种算生成失败，不再重跑

其他修改说明：
- `generate_with_cc_and_wiki.py`：调整算子生成任务 prompt 引导模型使用 `cann-ask` 技能查询知识；脚本默认放开知识查询 skill (`--allowed-tools Skill(cann-ask) Skill(setup-cann-wiki) Read Glob Grep`)


## 执行上板测试
`run_cc_static.sh` 可启动上板测试，需要注意按需调整`model_name`、`ASCEND_RT_VISIBLE_DEVICES`。
该脚本可以自动根据NPU类型调整算子代码，确保能够在本地NPU跑起来，原本测试脚本只能在A2上正常测试。

**注意事项**
evaluation后会在对应实验的算子代码目录下生成结果json（例如 `output/ascendc/add_shot/0.0-1.0/claude-code-static/run0/result_fuse.json`），查看可得知指定类型算子的编译和执行成功与否。

其他修改说明：
- `evaluation.py`：
    - 调整为针对例如fuse任务缺失的代码，直接算做生成失败，而不是报错退出评测
    - 调整为可以断点评测，否则中间意外中断要从头开始
