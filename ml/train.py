#!/usr/bin/env python3
"""
Train a small plant-disease classifier on a folder dataset, export ONNX for the API.

Expected folder layout (--data-dir):
  data/
    Tomato___Early_blight/*.jpg
    Tomato___Late_blight/*.jpg
    Tomato___healthy/*.jpg
    ...

Install (local only):
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r ml/requirements-train.txt

Run:
  python ml/train.py --data-dir ./data --epochs 5 --out-dir ./ml_out

Copy to API repo root for deploy:
  ml_out/plant_disease.onnx
  ml_out/labels.json
Set Render env: MODEL_PATH=plant_disease.onnx  (path relative to cwd, or absolute)

Tips:
  - Start with PlantVillage tomato subset (same folder names as class labels).
  - 20+ images per class minimum for a toy model; hundreds+ for real use.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out-dir", type=Path, default=Path("ml_out"))
    args = p.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"data dir not found: {args.data_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = datasets.ImageFolder(args.data_dir, transform=tfm)
    if len(ds.classes) < 2:
        raise SystemExit("Need at least 2 classes (subfolders) under --data-dir")

    # Save class order (must match ImageFolder order)
    labels_path = args.out_dir / "labels.json"
    labels_path.write_text(json.dumps({"classes": ds.classes}, indent=2), encoding="utf-8")
    print("Wrote", labels_path)

    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, len(ds.classes))

    m = m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        m.train()
        loss_tr = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = m(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_tr += loss.item() * x.size(0)
        loss_tr /= max(1, n_train)

        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = m(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(1, total)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={loss_tr:.4f}  val_acc={acc:.3f}")

    ckpt = args.out_dir / "plant_disease.pt"
    torch.save({"model": m.state_dict(), "classes": ds.classes}, ckpt)
    print("Wrote", ckpt)

    # Export ONNX (opset 17 works with onnxruntime on Render)
    m.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    import onnx as onnx_ir

    onnx_path = args.out_dir / "plant_disease.onnx"
    torch.onnx.export(
        m,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    # PyTorch may write a sibling .onnx.data file; Render only got the .onnx → load_error.
    # Merge external weights into one file and remove the .data sidecar.
    model_proto = onnx_ir.load(str(onnx_path), load_external_data=True)
    merged = args.out_dir / "plant_disease_merged.onnx"
    onnx_ir.save_model(model_proto, str(merged), save_as_external_data=False)
    merged.replace(onnx_path)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.is_file():
        sidecar.unlink()
    print("Wrote single-file ONNX", onnx_path)
    print("\nNext: copy plant_disease.onnx + labels.json next to main.py (or set MODEL_PATH).")
    print("Copy ml/advice_by_class.example.json to advice_by_class.json and edit for your classes.")


if __name__ == "__main__":
    main()
