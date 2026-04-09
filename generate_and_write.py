# Using LLM to generate code and output it to file
from utils.utils import get_client, get_google_client
from config import temperature, num_completions, max_tokens, top_p
from dataset import dataset, category2exampleop
import os
from prompt_generators.prompt_registry import PROMPT_REGISTRY 
from config import temperature, top_p
import importlib
import argparse

def generate_and_write_single(prompt, client, out_dir, op, model, max_retries=3):
    import time
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=temperature,
                n=num_completions,
                # max_tokens=max_tokens,
                top_p=top_p,
            )
            reasoning_content = ""  # 完整思考过程
            answer_content = ""  # 完整回复
            is_answering = False  # 是否进入回复阶段
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # 只收集思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                # 收到content，开始进行回复
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        is_answering = True
                    answer_content += delta.content
            if reasoning_content != '':
                with open(os.path.join(out_dir, f'{op}_cot.txt'), 'w') as out_file:
                    out_file.write(reasoning_content)
            with open(os.path.join(out_dir, f'{op}.txt'), 'w') as out_file:
                out_file.write(answer_content)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f"[WARN] Attempt {attempt+1} failed for {op}: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] All {max_retries} attempts failed for {op}: {e}")
                raise

GOOGLE_MODELS = ['gemma-4-31b-it']

def generate_and_write_single_google(prompt, client, out_dir, op, model, max_retries=5):
    import time
    from google.genai import types
    for attempt in range(max_retries):
        try:
            answer_content = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    system_instruction="You are a helpful assistant",
                ),
            ):
                if chunk.text:
                    answer_content += chunk.text
            with open(os.path.join(out_dir, f'{op}.txt'), 'w') as out_file:
                out_file.write(answer_content)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 10
                print(f"[WARN] Attempt {attempt+1} failed for {op}: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] All {max_retries} attempts failed for {op}: {e}")
                raise

def generate_prompt(language, strategy_name, op):
    if language not in PROMPT_REGISTRY or strategy_name not in PROMPT_REGISTRY[language]:
        try:
            importlib.import_module(f"prompt_generators.{language}_{strategy_name}")
        except ImportError as e:
            raise ValueError(f"Unsupported language/platform: {language} (module not found)") from e

    strategy = PROMPT_REGISTRY[language][strategy_name]
    return strategy.generate(op)

def generate_and_write(out_dir, language, model, op_tested, strategy):
    is_google = model in GOOGLE_MODELS
    if is_google:
        client = get_google_client()
    else:
        client = get_client(model)
    for i in range(len(op_tested)):
        op = op_tested[i]
        print(f'[INFO] Generate kernel for op {op}, strategy is {strategy}')
        if os.path.exists(os.path.join(out_dir, f'{op}.txt')):
            print(f"[INFO] Already generated at {out_dir}/{op}.txt, skip")
            continue
        prompt = generate_prompt(language, strategy, op)
        try:
            if is_google:
                generate_and_write_single_google(prompt, client, out_dir, op, model)
            else:
                generate_and_write_single(prompt, client, out_dir, op, model)
        except Exception as e:
            print(f"[ERROR] Skipping op {op}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model evaluation with specified parameters.")

    parser.add_argument('--runs', type=int, default=1, help='Number of runs.')
    parser.add_argument('--model', type=str, default='deepseek-chat', help='Model name.')
    parser.add_argument('--strategy', type=str, default='add_shot', help='Strategy type.')
    parser.add_argument('--categories', nargs='+', default=['activation'], help='List of categories.')

    args = parser.parse_args()

    runs = args.runs
    model = args.model
    language = 'ascendc'
    strategy = args.strategy
    categories = args.categories

    print(f"Runs: {runs}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Strategy: {strategy}")
    print(f"Categories: {categories}")

    op_tested = list(dataset.keys())
    if categories != ['all']:
        op_tested = [op for op in op_tested if dataset[op]['category'] in categories]

    if '/' in model:
        # handle slashed model names (e.g., provider/model-name)
        model_name = model.split('/')[1]
    else:
        model_name = model

    for run in range(runs):
        out_dir = f'output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}'
        os.makedirs(out_dir, exist_ok=True)
        generate_and_write(out_dir, language, model, op_tested, strategy)
    
    

