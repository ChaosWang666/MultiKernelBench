import os

# path
project_root_path = os.getcwd()
ref_impl_base_path = f'{project_root_path}/reference'

# trial
max_turn = 1
num_correct_trials = 5
num_perf_trials = 100
num_warmup = 3

# LLM config
max_tokens = 8192
temperature = 0.0
top_p=1.0
num_completions=1

seed_num=1024

# Ascend compile related
# OP_ENGINEER_DIR can override the workspace path (used by parallel workers for isolation)
op_engineer_dir = os.environ.get('OP_ENGINEER_DIR', f'{project_root_path}/ascend_op_projects')
deploy_path = f'{op_engineer_dir}/opp'
ascendc_device = 'ai_core-Ascend910B2'
