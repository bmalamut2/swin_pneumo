#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Swin + FPN segmentation on PNG pneumothorax dataset.

Folder layout:
.
├── png_images/                 # 0001.png, 0002.png, ...
├── png_masks/                  # 0001.png, 0002.png, ... (binary; >0 = foreground)
├── stage_1_train_images.csv    # column with image ids or filenames
└── stage_1_test_images.csv     # optional, for inference

Usage:
  python train_png_swin_seg.py --root . --curves_dir curves

What it logs each epoch:
  - train_total_loss = seg_loss + cls_loss_weight * cls_loss
  - train_seg_loss   = 0.5*BCE + 0.5*Dice (pixel-wise)
  - train_cls_loss   = image-level BCE ("pneumothorax present?")
  - val_total_loss, val_seg_loss, val_cls_loss
  - val_dice_pos     = Dice on positive validation images only
  - val_f1_cls       = image-level F1 on validation set

At the end it saves:
  - curves/loss_curve.png
  - curves/dice_curve.png
  - curves/training_history.csv
  - best weights: best_swin_fpn_png.pth  (name configurable)
"""

import os, random, gc, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryF1Score

# plotting (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ---------------------------
# Reproducibility
# ---------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# CSV helpers
# ---------------------------
def detect_id_column(df, candidates=("ImageId","image_id","filename","file_name","id","image","img","name","path")):
    for c in candidates:
        if c in df.columns:
            print(f"[info] Using ID column: '{c}'")
            return c
    c = df.columns[0]
    print(f"[warn] None of {candidates} found; using first column: '{c}'")
    return c


# ---------------------------
# IO / preprocessing
# ---------------------------
def ensure_gray_float01(img):
    """Convert any PNG (uint8/uint16, gray/RGB) to float32 in [0,1] with robust contrast."""
    if img is None:
        raise ValueError("Failed to read image.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, (0.5, 99.5))
    if p_high <= p_low:
        mn, mx = float(img.min()), float(img.max())
        img = (img - mn) / (mx - mn + 1e-6) if mx > mn else np.zeros_like(img)
    else:
        img = np.clip((img - p_low) / (p_high - p_low + 1e-6), 0, 1)
    return img

def read_png_mask(mask_path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = (m > 0).astype(np.uint8)   # binarize
    return m

def has_positive_mask(mask_path):
    m = read_png_mask(mask_path)
    return (m is not None) and (m.any())


# ---------------------------
# Dataset & index
# ---------------------------
class PneumoPNGs(Dataset):
    def __init__(self, table, img_size=768, augment=False):
        self.table = table.reset_index(drop=True)
        self.img_size = img_size
        if augment:
            self.tf = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.15, rotate_limit=12, p=0.7, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomBrightnessContrast(p=0.3),
                A.CoarseDropout(max_holes=4, max_height=int(0.05*img_size), max_width=int(0.05*img_size), p=0.2),
                ToTensorV2()
            ])
        else:
            self.tf = A.Compose([A.Resize(img_size, img_size), ToTensorV2()])

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        r = self.table.iloc[idx]
        img = cv2.imread(r["image_path"], cv2.IMREAD_UNCHANGED)
        img = ensure_gray_float01(img)  # HxW float32 [0,1]

        # read mask if exists
        mask = read_png_mask(r["mask_path"]) if r["mask_exists"] else None
        if mask is None:
            mask = np.zeros_like(img, dtype=np.uint8)

        data = self.tf(image=(img*255).astype(np.uint8), mask=mask.astype(np.uint8))
        x = data["image"].float() / 255.0           # 1xHxW
        y = data["mask"].unsqueeze(0).float()       # 1xHxW
        y_cls = torch.tensor([1.0 if r["has_mask"] else 0.0], dtype=torch.float32)
        return x, y, y_cls


def build_index_from_csv(root, train_csv, img_dir, mask_dir):
    df = pd.read_csv(train_csv)
    id_col = detect_id_column(df)
    rows = []
    for _, row in df.iterrows():
        name = str(row[id_col])
        stem = Path(name).stem
        image_path = str(Path(img_dir) / f"{stem}.png")
        mask_path  = str(Path(mask_dir) / f"{stem}.png")
        exists_m = Path(mask_path).exists()
        pos = has_positive_mask(mask_path) if exists_m else False
        rows.append({
            "image_id": stem,
            "image_path": image_path,
            "mask_path": mask_path,
            "mask_exists": exists_m,
            "has_mask": pos
        })
    table = pd.DataFrame(rows)
    keep = table["image_path"].apply(lambda p: Path(p).exists())
    missing = (~keep).sum()
    if missing:
        print(f"[warn] {missing} images from CSV not found in '{img_dir}'. They will be ignored.")
    table = table[keep].reset_index(drop=True)
    print(f"[info] Indexed {len(table)} images | positives={table['has_mask'].sum()} | "
          f"negatives={len(table) - table['has_mask'].sum()}")
    return table


# ---------------------------
# Model: Swin encoder + FPN decoder (with aux classification head)
# ---------------------------
class SMPWithAux(nn.Module):
    """
    Wrapper that returns (seg_logits, cls_logit).
    Uses FPN decoder for broad compatibility in segmentation_models_pytorch.
    """
    def __init__(self, encoder_name="timm-swin_tiny_patch4_window7_224", weights="imagenet"):
        super().__init__()
        aux_params = dict(pooling="avg", dropout=0.25, classes=1, activation=None)
        # Try the provided encoder_name; if it fails due to naming differences, try swapping timm<->tu
        try:
            self.net = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=weights,
                in_channels=1,
                classes=1,
                aux_params=aux_params
            )
        except Exception as e:
            # Attempt fallback between "timm-" and "tu-"
            if encoder_name.startswith("timm-"):
                alt_name = encoder_name.replace("timm-", "tu-")
            elif encoder_name.startswith("tu-"):
                alt_name = encoder_name.replace("tu-", "timm-")
            else:
                alt_name = None
            if alt_name:
                print(f"[warn] encoder '{encoder_name}' failed ({e}). Trying '{alt_name}'...")
                self.net = smp.FPN(
                    encoder_name=alt_name,
                    encoder_weights=weights,
                    in_channels=1,
                    classes=1,
                    aux_params=aux_params
                )
            else:
                raise

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            seg, cls = out
        else:
            seg = out
            cls = seg.mean(dim=[2,3], keepdim=False)   # fallback proxy
        return seg, cls


# ---------------------------
# Losses & metrics
# ---------------------------
bce = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="binary")

def seg_loss(pred_mask, true_mask):
    return 0.5 * bce(pred_mask, true_mask) + 0.5 * dice_loss(pred_mask, true_mask)


# ---------------------------
# Sampler & splits
# ---------------------------
def make_sampler(table, pos_ratio=0.5):
    y = table["has_mask"].astype(int).values
    pos_w = pos_ratio / max(y.sum(), 1)
    neg_w = (1 - pos_ratio) / max((y == 0).sum(), 1)
    weights = np.where(y == 1, pos_w, neg_w).astype(np.float64)
    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def split_table(tbl, val_frac=0.1, seed=13):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(tbl))
    rng.shuffle(idx)
    v = int(len(tbl) * val_frac)
    val_idx = idx[:v]
    tr_idx = idx[v:]
    return tbl.iloc[tr_idx].reset_index(drop=True), tbl.iloc[val_idx].reset_index(drop=True)


# ---------------------------
# Evaluation (now returns losses for plotting)
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, device, cls_loss_weight):
    model.eval()
    dices, f1s = [], []
    n_pos = 0
    val_seg_losses, val_cls_losses = [], []

    f1_metric = BinaryF1Score(threshold=0.5).to(device)

    for x, y, y_cls in loader:
        x, y, y_cls = x.to(device), y.to(device), y_cls.to(device)
        logits, cls_logit = model(x)

        # losses
        seg_l = seg_loss(logits, y).item()
        cls_l = bce(cls_logit, y_cls).item()
        val_seg_losses.append(seg_l)
        val_cls_losses.append(cls_l)

        # dice on positives only
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        pos_mask = (y.sum(dim=[1,2,3]) > 0)
        if pos_mask.any():
            inter = (pred[pos_mask] * y[pos_mask]).sum(dim=[1,2,3])
            den = pred[pos_mask].sum(dim=[1,2,3]) + y[pos_mask].sum(dim=[1,2,3])
            dice = (2*inter + 1e-6) / (den + 1e-6)
            dices.extend(dice.detach().cpu().numpy().tolist())
            n_pos += int(pos_mask.sum())

        # image-level F1
        f1s.append(f1_metric(torch.sigmoid(cls_logit).squeeze(1), y_cls.squeeze(1)))

    mean_seg = float(np.mean(val_seg_losses)) if val_seg_losses else 0.0
    mean_cls = float(np.mean(val_cls_losses)) if val_cls_losses else 0.0
    mean_total = mean_seg + cls_loss_weight * mean_cls
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_f1 = float(torch.stack(f1s).mean().item())

    return dict(
        dice_pos=mean_dice,
        f1_cls=mean_f1,
        n_pos=n_pos,
        val_seg_loss=mean_seg,
        val_cls_loss=mean_cls,
        val_total_loss=mean_total,
    )


# ---------------------------
# Training
# ---------------------------
def train(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.root)
    img_dir = root / "png_images"
    mask_dir = root / "png_masks"
    train_csv = root / "stage_1_train_images.csv"
    test_csv  = root / "stage_1_test_images.csv"  # optional / unused in training

    curves_dir = root / args.curves_dir
    curves_dir.mkdir(parents=True, exist_ok=True)

    table = build_index_from_csv(root, train_csv, img_dir, mask_dir)
    train_tbl, val_tbl = split_table(table, val_frac=args.val_frac, seed=args.seed)

    train_ds = PneumoPNGs(train_tbl, img_size=args.img_size, augment=True)
    val_ds   = PneumoPNGs(val_tbl,   img_size=args.img_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train_tbl, pos_ratio=args.pos_ratio),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    model = SMPWithAux(encoder_name=args.encoder, weights=args.encoder_weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # --- Track metrics across epochs ---
    history = {
        "epoch": [],
        "train_total_loss": [],
        "train_seg_loss": [],
        "train_cls_loss": [],
        "val_total_loss": [],
        "val_seg_loss": [],
        "val_cls_loss": [],
        "val_dice_pos": [],
        "val_f1_cls": [],
    }

    best = {"dice_pos": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_seg, run_cls, run_total, n_batches = 0.0, 0.0, 0.0, 0

        for x, y, y_cls in train_loader:
            x, y, y_cls = x.to(device), y.to(device), y_cls.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast():
                logits, cls_logit = model(x)
                loss_seg = seg_loss(logits, y)
                loss_cls = bce(cls_logit, y_cls)
                loss = loss_seg + args.cls_loss_weight * loss_cls
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_seg += loss_seg.item()
            run_cls += loss_cls.item()
            run_total += loss.item()
            n_batches += 1

        # Validation (also returns losses for curves)
        val_metrics = evaluate(model, val_loader, device, cls_loss_weight=args.cls_loss_weight)

        # Console log
        print(
            f"Epoch {epoch:02d} | "
            f"train_total={run_total/max(1,n_batches):.4f} "
            f"train_seg={run_seg/max(1,n_batches):.4f} "
            f"train_cls={run_cls/max(1,n_batches):.4f} | "
            f"val_total={val_metrics['val_total_loss']:.4f} "
            f"val_seg={val_metrics['val_seg_loss']:.4f} "
            f"val_cls={val_metrics['val_cls_loss']:.4f} | "
            f"val_dice_pos={val_metrics['dice_pos']:.4f} "
            f"val_f1_cls={val_metrics['f1_cls']:.4f} "
            f"(n_pos={val_metrics['n_pos']})"
        )

        # --- Update history ---
        history["epoch"].append(epoch)
        history["train_total_loss"].append(run_total / max(1, n_batches))
        history["train_seg_loss"].append(run_seg / max(1, n_batches))
        history["train_cls_loss"].append(run_cls / max(1, n_batches))
        history["val_total_loss"].append(val_metrics["val_total_loss"])
        history["val_seg_loss"].append(val_metrics["val_seg_loss"])
        history["val_cls_loss"].append(val_metrics["val_cls_loss"])
        history["val_dice_pos"].append(val_metrics["dice_pos"])
        history["val_f1_cls"].append(val_metrics["f1_cls"])

        # Save best by validation dice on positives
        if val_metrics["dice_pos"] > best["dice_pos"]:
            best.update(val_metrics)
            save_path = root / args.out_weights
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, save_path)
            print(f"[info] Saved best to: {save_path}")

    print(f"[done] Best positive-case Dice: {best['dice_pos']:.4f}")

    # --- Persist history ---
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(curves_dir / "training_history.csv", index=False)

    # --- Plot 1: Loss over epochs (train vs val total loss) ---
    plt.figure()
    plt.plot(history["epoch"], history["train_total_loss"], label="Train total loss")
    plt.plot(history["epoch"], history["val_total_loss"], label="Val total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(curves_dir / "loss_curve.png", dpi=150)
    plt.close()

    # --- Plot 2: Dice over epochs (validation, positives only) ---
    plt.figure()
    plt.plot(history["epoch"], history["val_dice_pos"], label="Val Dice (positives only)")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice over epochs")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(curves_dir / "dice_curve.png", dpi=150)
    plt.close()

    print(f"[curves] Saved: {curves_dir / 'loss_curve.png'}")
    print(f"[curves] Saved: {curves_dir / 'dice_curve.png'}")
    print(f"[curves] History CSV: {curves_dir / 'training_history.csv'}")


# ---------------------------
# Optional: simple inference to PNGs
# ---------------------------
@torch.no_grad()
def predict_to_pngs(args, weights_path, out_dir="pred_masks"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.root)
    img_dir = root / "png_images"
    test_csv = root / "stage_1_test_images.csv"
    out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv)
    id_col = detect_id_column(df)

    model = SMPWithAux(encoder_name=args.encoder, weights=args.encoder_weights).to(device)
    state = torch.load(Path(weights_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    tf = A.Compose([A.Resize(args.img_size, args.img_size), ToTensorV2()])

    for _, row in df.iterrows():
        stem = Path(str(row[id_col])).stem
        ipath = img_dir / f"{stem}.png"
        img = cv2.imread(str(ipath), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[warn] missing image: {ipath}")
            continue
        img = ensure_gray_float01(img)
        data = tf(image=(img*255).astype(np.uint8))
        x = data["image"].float().unsqueeze(0) / 255.0  # 1x1xHxW
        x = x.to(device)
        logits, _ = model(x)
        prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()
        mask = (prob > 0.5).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{stem}.png"), mask)


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="dataset root with png_images/, png_masks/, CSVs")
    ap.add_argument("--encoder", type=str, default="timm-swin_tiny_patch4_window7_224",
                    help="Swin backbone name. If your SMP expects 'tu-*', pass 'tu-swin_tiny_patch4_window7_224'.")
    ap.add_argument("--encoder_weights", type=str, default="imagenet")
    ap.add_argument("--img_size", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pos_ratio", type=float, default=0.5, help="target positive fraction in sampler")
    ap.add_argument("--cls_loss_weight", type=float, default=0.2, help="weight for image-level BCE loss")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_weights", type=str, default="best_swin_fpn_png.pth")
    ap.add_argument("--curves_dir", type=str, default="curves",
                    help="where to save training curves and history CSV")
    ap.add_argument("--predict_only", action="store_true", help="skip training and run inference to PNGs")
    ap.add_argument("--weights_for_predict", type=str, default="best_swin_fpn_png.pth")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.predict_only:
        predict_to_pngs(args, weights_path=Path(args.root)/args.weights_for_predict)
    else:
        train(args)
