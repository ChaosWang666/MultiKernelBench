"""ascendc anti-hack prompt strategy (默认).

- 模板使用 ascendc_anti_hack_template (含 ANTI-HACK RULES + COUNTER-EXAMPLES + L1 schema 嵌入)
- one-shot 算子 gemm_multiply_leakyrelu (不在 selected_ops.yaml 评测集, 避免数据污染;
  含 nn.Linear weight + nn.Parameter bias + 标量 init_attr, 完整演示新 ModelNew 形态)

注册了两个 strategy 名 (同一个实例):
- 'anti_hack' — 主名, 文档主推
- 'add_shot'  — 旧 strategy 名, 保留向后兼容 (evaluation.py / generate_and_write.py
                的默认 --strategy add_shot 调用自动走加固版)
"""
from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from prompt_generators.prompt_utils import (
    read_relavant_files, ascendc_anti_hack_template,
)


@register_prompt("ascendc", "anti_hack")
@register_prompt("ascendc", "add_shot")
class AscendcAntiHackStrategy(BasePromptStrategy):
    EXAMPLE_OP = 'gemm_multiply_leakyrelu'

    def generate(self, op):
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files(
            'ascendc', op, self.EXAMPLE_OP)
        return ascendc_anti_hack_template(
            arc_src, example_arch_src, example_new_arch_src, op, self.EXAMPLE_OP)
