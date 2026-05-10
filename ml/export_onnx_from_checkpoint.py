#!/usr/bin/env python3
"""
Rebuild a single-file plant_disease.onnx from ml_out/plant_disease.pt (no training).

Use when your .onnx points at a missing .onnx.data (common after git push).
Requires: pip install -r ml/requirements-train.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx as onnx_ir
import torch
import torch.nn as nn
from torchvision import models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("ml_out/plant_disease.pt"))
    p.add_argument("--out", type=Path, default=Path("plant_disease.onnx"))
    args = p.parse_args()

    if not args.ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes: list[str] = ckpt["classes"]
    state = ckpt["model"]

    m = models.mobilenet_v3_small(weights=None)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, len(classes))
    m.load_state_dict(state)
    m.eval()

    dummy = torch.randn(1, 3, 224, 224)
    tmp = args.out.with_suffix(".export_tmp.onnx")
    torch.onnx.export(
        m,
        dummy,
        tmp,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    model_proto = onnx_ir.load(str(tmp), load_external_data=True)
    onnx_ir.save_model(model_proto, str(args.out), save_as_external_data=False)
    tmp.unlink(missing_ok=True)
    sidecar = tmp.with_name(tmp.name + ".data")
    sidecar.unlink(missing_ok=True)
    print("Wrote", args.out.resolve())


if __name__ == "__main__":
    main()
