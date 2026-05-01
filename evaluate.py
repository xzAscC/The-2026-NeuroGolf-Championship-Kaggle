#!/usr/bin/env python3
"""Local evaluator for NeuroGolf 2026 submissions.

Uses onnx_tool (the same profiler the Kaggle scorer uses) to compute:
  score = max(1, 25 - ln(cost))
  cost  = params + memory_bytes + MACs

Usage:
  python evaluate.py --submission_dir submission --data_file data/all_tasks.json
  python evaluate.py --submission_dir submission
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import onnx
import onnx_tool
import onnxruntime as ort

FILE_SIZE_LIMIT = 1_474_560  # 1.44 MB


def to_onehot(grid, ch=10, gh=30, gw=30):
    arr = np.zeros([1, ch, gh, gw], dtype=np.float32)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if 0 <= int(v) < ch:
                arr[0, int(v), r, c] = 1.0
    return arr


def check_correctness(model_path, task_data):
    try:
        sess = ort.InferenceSession(model_path)
        inp_name = sess.get_inputs()[0].name
        for pair in task_data["train"] + task_data["test"]:
            inp = to_onehot(pair["input"])
            expected = to_onehot(pair["output"])
            out = sess.run(None, {inp_name: inp})[0]
            out = (out > 0.0).astype(np.float32)
            if not np.array_equal(out, expected):
                return False
        return True
    except Exception as e:
        print(f"    RUNTIME ERROR: {e}", file=sys.stderr)
        return False


def score_model(model_path):
    file_size = os.path.getsize(model_path)
    if file_size > FILE_SIZE_LIMIT:
        return 0, 0, 0, 0, 0.0, file_size, False, "file too large"

    model = onnx_tool.loadmodel(str(model_path), {"verbose": False})
    graph = model.graph
    graph.graph_reorder_nodes()
    graph.shape_infer(None)
    graph.profile()

    macs = int(sum(graph.macs))
    memory_bytes = int(graph.memory)
    params = int(graph.params)
    cost = macs + memory_bytes + params
    score = max(1.0, 25.0 - math.log(cost)) if cost > 0 else 25.0
    return macs, memory_bytes, params, cost, score, file_size, True, ""


def main():
    parser = argparse.ArgumentParser(description="NeuroGolf 2026 Local Evaluator")
    parser.add_argument("--submission_dir", default="submission")
    parser.add_argument("--data_file", default=None)
    args = parser.parse_args()

    tasks = {}
    if args.data_file and os.path.exists(args.data_file):
        with open(args.data_file) as f:
            raw = json.load(f)
        for idx, key in enumerate(sorted(raw.keys())):
            tasks[idx] = raw[key]

    files = sorted(f for f in os.listdir(args.submission_dir) if f.endswith(".onnx"))
    if not files:
        print(f"No ONNX files found in {args.submission_dir}/")
        sys.exit(1)

    total_score = 0
    correct_count = 0
    results = []

    for fn in files:
        path = os.path.join(args.submission_dir, fn)
        task_num = int(fn.replace("task_", "").replace(".onnx", ""))

        macs, mem, params, cost, score, fsize, valid, reason = score_model(path)

        correct = None
        if valid and tasks and task_num in tasks:
            correct = check_correctness(path, tasks[task_num])
            if correct:
                correct_count += 1

        contributes = (correct is None or correct) and valid
        total_score += score if contributes else 0
        results.append(
            (fn, params, mem, macs, cost, score, fsize, correct, valid, reason)
        )

    print(
        f"{'Task':<18} {'Params':>7} {'Mem':>9} {'MACs':>9} {'Cost':>10} {'Score':>7} {'Size':>7} {'OK':>5}"
    )
    print("-" * 82)
    for fn, params, mem, macs, cost, score, fsize, correct, valid, reason in results:
        if not valid:
            ok = f"BAD({reason})"
        elif correct is None:
            ok = "?"
        elif correct:
            ok = "Y"
        else:
            ok = "FAIL"
        sz = f"{fsize / 1024:.0f}K"
        print(
            f"{fn:<18} {params:>7} {mem:>9} {macs:>9} {cost:>10} {score:>7.3f} {sz:>7} {ok:>5}"
        )

    print("-" * 82)
    n_valid = sum(1 for r in results if (r[7] is None or r[7]) and r[8])
    n_correct = sum(1 for r in results if r[7] is True)
    print(f"Files: {len(files)}  Scored: {n_valid}  Correct: {n_correct}")
    print(f"Total score: {total_score:.2f}")

    if tasks:
        n_total = len(tasks)
        pct = n_correct / n_total * 100
        print(f"Coverage: {n_correct}/{n_total} ({pct:.1f}%)")

    best = sorted(results, key=lambda r: -r[5])[:5]
    print(f"\nBest scoring tasks:")
    for fn, _, _, _, _, score, _, _, _, _ in best:
        print(f"  {fn}: {score:.3f}")

    worst = sorted(results, key=lambda r: r[5])[:5]
    print(f"Worst scoring tasks:")
    for fn, _, _, _, _, score, _, _, _, _ in worst:
        print(f"  {fn}: {score:.3f}")


if __name__ == "__main__":
    main()
