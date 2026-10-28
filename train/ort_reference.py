"""Write the deterministic reference sample for checking the C# sampler (tools/csharp_check, in-game dev action).

Runs export_onnx.ort_sample (numpy + ONNX Runtime, argmax, no noise) on one grid and writes it as digits,
one line per row, row 0 (south) first - the same format as the C# side writes.

Usage (CPU is enough): python -u train/ort_reference.py Models/v1_mdm [out.txt]
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_onnx import ort_sample  # noqa: E402


def main():
    model_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(model_dir, "ref_deterministic.txt")
    meta = json.load(open(os.path.join(model_dir, "model_meta.json")))
    sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), providers=["CPUExecutionProvider"])
    n = meta["window"]
    grid = ort_sample(sess, 1, (n, n), meta["num_classes"], meta["sampler"]["steps"],
                      np.array(meta["logit_bias"], np.float32))[0]
    with open(out, "w", newline="\n") as f:
        f.write("".join("".join(str(int(v)) for v in row) + "\n" for row in grid))
    print(f"wrote {out} (ONNX Runtime {ort.__version__})")


if __name__ == "__main__":
    main()
