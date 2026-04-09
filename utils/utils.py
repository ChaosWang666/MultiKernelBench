import os
import config
import re
from openai import OpenAI
from dataset import dataset
import torch

# api3.xhub.chat平台
def get_client(model):
    client = OpenAI(
        api_key="sk-qLGf3r2HFjyUrtvwqWUkzY6m9LIFOJr2xleMMAO04t6nJceY",
        base_url="https://api3.xhub.chat/v1",
        timeout=10000000,
        max_retries=3,
    )
    return client

# 阿里云平台
def get_client(model):
    client = OpenAI(
        api_key="sk-ece3d4b1280243b2acb266002a3e3fd0",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=10000000,
        max_retries=3,
    )
    return client



# Google genai
def get_google_client():
    from google import genai
    from google.genai import types
    import httpx
    http_client = httpx.Client(proxy="http://127.0.0.1:7899", timeout=httpx.Timeout(600))
    client = genai.Client(
        api_key="AIzaSyCXzUusLG1zQzLOopfaoZabA47Lr99Yf1k",
        http_options=types.HttpOptions(
            httpx_client=http_client,
            timeout=600000,
        ),
    )
    return client

def get_ref_src_path(op):
    return os.path.join(config.ref_impl_base_path, dataset[op]['category'], f'{op}.py')


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code_block = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code_block.startswith(code_type):
                code = code_block[len(code_type) :].strip()

        return code, f'```{code_block}```'

def underscore_to_pascalcase(underscore_str):
    """
    Convert underscore-separated string to PascalCase.
    
    Args:
        underscore_str (str): Input string with underscores (e.g., "vector_add")
        
    Returns:
        str: PascalCase version (e.g., "VectorAdd")
    """
    if not underscore_str:  # Handle empty string
        return ""
    
    parts = underscore_str.split('_')
    # Capitalize the first letter of each part and join
    return ''.join(word.capitalize() for word in parts if word)