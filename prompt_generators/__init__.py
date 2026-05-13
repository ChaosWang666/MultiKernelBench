"""Eager-import all strategy modules so their @register_prompt decorators run
at package import time.

Without this, `generate_prompt(language, strategy_name, op)` in
generate_and_write.py falls back to `importlib.import_module(
    f"prompt_generators.{language}_{strategy_name}")`, which fails for
strategy aliases whose module filename differs from the strategy name
(e.g. strategy 'add_shot' lives in ascendc_anti_hack.py).
"""
from prompt_generators import ascendc_anti_hack  # registers 'anti_hack' + 'add_shot'
from prompt_generators import ascendc_selected_shot  # registers 'selected_shot'

__all__ = ["ascendc_anti_hack", "ascendc_selected_shot"]
