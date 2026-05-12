#!/usr/bin/env python3
"""
Train a plant-disease classifier (ImageFolder), export single-file ONNX for the API.

Expected layout (--data-dir):
  data/Tomato___Early_blight/*.jpg
      Tomato___Late_blight/*.jpg
      ...

Install:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r ml/requirements-train.txt

Run (GPU / Colab T4 friendly):
  python ml/train.py --data-dir ./data --epochs 25 --batch-size 32 --lr 3e-4 --out-dir ./ml_out

Outputs:
  ml_out/plant_disease.onnx  (single file; merged external weights if any)
  ml_out/labels.json
  ml_out/plant_disease.pt    (best validation weights + metadata)

API deploy: MODEL_PATH=plant_disease.onnx, LABELS_PATH=labels.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_val_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def stratified_indices(
    dataset: datasets.ImageFolder,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split indices per class so val set is roughly balanced across classes."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {c: [] for c in range(len(dataset.classes))}
    for idx, (_, y) in enumerate(dataset.samples):
        by_class[y].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for _cls, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 0:
            continue
        if n == 1:
            train_idx.extend(idxs)
            continue
        n_val = max(1, int(round(n * val_fraction)))
        if n_val >= n:
            n_val = n - 1
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    if not train_idx:
        raise SystemExit("Stratified split produced empty train set; add more images per class.")
    if not val_idx:
        raise SystemExit("Stratified split produced empty val set; add more images per class or lower --val-fraction.")

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_model(backbone: Literal["efficientnet_b0", "mobilenet_v3_small"], num_classes: int) -> nn.Module:
    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, num_classes)
    return m


def backbone_param_prefixes(backbone: str) -> tuple[str, ...]:
    if backbone == "efficientnet_b0":
        return ("features.",)
    return ("features.",)


def is_backbone_param(name: str, backbone: str) -> bool:
    return name.startswith(backbone_param_prefixes(backbone))


def merge_onnx_single_file(onnx_path: Path) -> None:
    import onnx as onnx_ir

    proto = onnx_ir.load(str(onnx_path), load_external_data=True)
    merged = onnx_path.with_suffix(".merged.onnx")
    onnx_ir.save_model(proto, str(merged), save_as_external_data=False)
    merged.replace(onnx_path)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.is_file():
        sidecar.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description="Train plant disease classifier → ONNX")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4, help="Peak LR for classifier head (backbone uses --lr-backbone-mult)")
    p.add_argument("--lr-backbone-mult", type=float, default=0.15, help="Backbone LR = lr * this factor")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.08)
    p.add_argument("--val-fraction", type=float, default=0.12, help="Fraction held out per class (stratified)")
    p.add_argument("--early-stop-patience", type=int, default=8, help="Stop if val acc does not improve for N epochs")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--backbone",
        choices=("efficientnet_b0", "mobilenet_v3_small"),
        default="efficientnet_b0",
        help="efficientnet_b0 = stronger; mobilenet_v3_small = smaller / faster",
    )
    p.add_argument("--out-dir", type=Path, default=Path("ml_out"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = auto: up to 4 on CUDA, capped by CPU count; else 0)",
    )
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision even on CUDA")
    args = p.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"data dir not found: {args.data_dir}")

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    use_amp = device.type == "cuda" and not args.no_amp
    workers = args.workers if args.workers >= 0 else (4 if device.type == "cuda" else 0)
    if workers > 0:
        cpu = os.cpu_count() or 2
        workers = min(workers, max(1, cpu - 1))
    pin = device.type == "cuda"

    # Two ImageFolders = correct train vs val augmentations (no shared-transform bug)
    train_ds_full = datasets.ImageFolder(args.data_dir, transform=build_train_transforms())
    val_ds_full = datasets.ImageFolder(args.data_dir, transform=build_val_transforms())

    if len(train_ds_full.classes) < 2:
        raise SystemExit("Need at least 2 classes (subfolders) under --data-dir")

    classes = train_ds_full.classes
    labels_path = args.out_dir / "labels.json"
    labels_path.write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    print("Wrote", labels_path, flush=True)

    train_idx, val_idx = stratified_indices(train_ds_full, args.val_fraction, args.seed)
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)

    print(f"Train images: {len(train_idx)} | Val images: {len(val_idx)} | Classes: {len(classes)}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        # Avoid empty loader when train set is smaller than two batches
        drop_last=len(train_idx) >= 2 * args.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
    )

    model = build_model(args.backbone, len(classes)).to(device)

    backbone_lr = args.lr * args.lr_backbone_mult
    param_groups: list[dict] = []
    back_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_backbone_param(name, args.backbone):
            back_params.append(param)
        else:
            head_params.append(param)
    if back_params:
        param_groups.append({"params": back_params, "lr": backbone_lr})
    param_groups.append({"params": head_params, "lr": args.lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    steps_per_epoch = max(1, len(train_loader))
    max_lrs = [g["lr"] for g in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler = amp.GradScaler("cuda", enabled=use_amp)

    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = args.early_stop_patience

    for epoch in range(args.epochs):
        model.train()
        loss_tr = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = x.size(0)
            loss_tr += loss.item() * bs
            n_seen += bs

        train_loss = loss_tr / max(1, n_seen)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=pin)
                y = y.to(device, non_blocking=pin)
                with amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        acc = correct / max(1, total)
        print(
            f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={acc:.4f}  "
            f"lr_head={optimizer.param_groups[-1]['lr']:.2e}",
            flush=True,
        )

        if acc > best_acc + 1e-6:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping (no val improvement for {args.early_stop_patience} epochs).", flush=True)
                break

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()

    ckpt = args.out_dir / "plant_disease.pt"
    torch.save({"model": best_state, "classes": classes, "val_acc": best_acc, "backbone": args.backbone}, ckpt)
    print(f"Wrote {ckpt} (best val_acc={best_acc:.4f})", flush=True)

    # ONNX: export float32 on CPU for widest onnxruntime / API compatibility
    onnx_path = args.out_dir / "plant_disease.onnx"
    export_cpu = model.cpu().eval().float()
    dummy_cpu = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        export_cpu,
        dummy_cpu,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    try:
        merge_onnx_single_file(onnx_path)
        print("Wrote single-file ONNX", onnx_path, flush=True)
    except Exception as e:
        print(f"ONNX merge warning (non-fatal): {e}", file=sys.stderr)

    print("\nNext: copy plant_disease.onnx + labels.json next to main.py (or set MODEL_PATH / LABELS_PATH).", flush=True)
    print("Copy ml/advice_by_class.example.json → advice_by_class.json and align keys with labels.json classes.", flush=True)


if __name__ == "__main__":
    main()
