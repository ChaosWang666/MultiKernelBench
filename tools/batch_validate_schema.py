"""批量跑 selected_ops.yaml 里全部算子的 schema 提取,定位边界 case。

每个算子起独立 subprocess,避免内存累积/OOM (有些 reference 的 input 单个就 1GB)。

用法:
    cd MultiKernelBench
    python tools/batch_validate_schema.py [<selected_ops.yaml>]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
MKB_ROOT = THIS_DIR.parent


def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else \
        str(MKB_ROOT / "selected_ops.yaml")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    base_dir = MKB_ROOT / "reference"
    ops = data["ops"]
    print(f"validating {len(ops)} ops from {yaml_path}")
    print(f"base_dir = {base_dir}")
    print(f"runner   = {sys.executable} (subprocess per op)\n")

    ok, fail = [], []
    for i, op in enumerate(ops, 1):
        name = op["name"]
        category = op["category"]
        rel = op["path"]
        full = base_dir / rel

        # 子进程跑,JSON 输出
        cmd = [sys.executable, str(MKB_ROOT / "utils" / "schema_extractor.py"),
               str(full), name, category, "--json"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=180, cwd=str(MKB_ROOT))
        except subprocess.TimeoutExpired:
            print(f"FAIL  [{category:>13}] {name:55s} TIMEOUT(>180s)")
            fail.append((name, category, "timeout", ""))
            continue

        if p.returncode != 0:
            err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else f"exit={p.returncode}"
            print(f"FAIL  [{category:>13}] {name:55s} {err}")
            fail.append((name, category, err, p.stderr))
            continue

        try:
            schema = json.loads(p.stdout)
        except json.JSONDecodeError as e:
            print(f"FAIL  [{category:>13}] {name:55s} JSON parse: {e}")
            fail.append((name, category, str(e), p.stdout[-500:]))
            continue

        n_in  = len(schema.get("inputs", []))
        n_p   = len(schema.get("parameters", []))
        n_b   = len(schema.get("buffers", []))
        n_a   = len(schema.get("init_attrs", []))
        n_out = len(schema.get("outputs", []))
        ok.append((name, schema))
        print(f"  OK  [{category:>13}] {name:55s} "
              f"in={n_in} param={n_p} buf={n_b} attr={n_a} out={n_out}")

    print(f"\n=== summary: {len(ok)}/{len(ops)} OK, {len(fail)} FAIL ===")
    if fail:
        print("\n--- failures ---")
        for name, cat, err, detail in fail:
            print(f"\n### {cat}/{name}")
            print(f"error: {err}")
            if detail:
                # 只打末尾几行 stderr
                tail = "\n".join(detail.strip().splitlines()[-15:])
                print(tail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
