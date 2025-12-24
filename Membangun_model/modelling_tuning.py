""" 
Kriteria 2 - Skilled (3 pts) & Advanced (4 pts - DagsHub)

Skilled (3 pts)
- Kriteria Basic wajib terpenuhi
- Training + hyperparameter tuning
- WAJIB manual logging (bukan autolog)
- Metrics minimal sama seperti autolog (accuracy, precision, recall, f1)

Advanced (4 pts)
- Tracking online via DagsHub (opsional via flag)

Implementasi: CNN transfer learning
- Pretrained ResNet18 (ImageNet) sebagai backbone
- Layer backbone dibekukan, lalu dilatih (transfer learning)
- Hyperparameter tuning sederhana: grid kecil untuk learning rate & weight decay
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import matplotlib.pyplot as plt
import seaborn as sns


DOTENV_PATH_LOADED: str | None = None


def load_dotenv_if_available() -> None:
    """Load environment variables from a .env file if python-dotenv is available.

    Note: A .env file is NOT automatically read by Python or PowerShell.
    We explicitly load it so flags like --use_dagshub can read DAGSHUB_*.
    """

    global DOTENV_PATH_LOADED
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore

        # Search from current working directory upwards.
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)
            DOTENV_PATH_LOADED = dotenv_path
    except Exception:
        # If python-dotenv isn't installed or anything goes wrong,
        # we just fall back to real OS environment variables.
        DOTENV_PATH_LOADED = None


# Load .env ASAP so argparse defaults using os.getenv() can see it.
load_dotenv_if_available()


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Iterable[str], labels: Iterable[int], transform):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label = int(self.labels[idx])
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skilled/Advanced: CNN transfer learning (ResNet18) + tuning + manual MLflow logging"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Vegetables_preprocessing",
        help="Folder berisi train_ready.csv/test_ready.csv (default: Vegetables_preprocessing)",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=str(Path("..") / "Eksperimen_MSML_Daffa_Al-Fathir_Ismail" / "Vegetables_raw"),
        help="Folder Vegetables_raw untuk fallback path (default: ../Eksperimen_.../Vegetables_raw)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Vegetable Classification Skilled (Tuning)",
        help="Nama MLflow experiment",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size untuk feature extraction",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=0,
        help="Batasi jumlah data train agar cepat (0 = semua / pakai seluruh data)",
    )
    parser.add_argument(
        "--max_test",
        type=int,
        default=0,
        help="Batasi jumlah data test agar cepat (0 = semua / pakai seluruh data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Epoch untuk setiap konfigurasi hyperparameter",
    )
    parser.add_argument(
        "--lr_grid",
        type=str,
        default="0.001,0.0003",
        help="Grid learning rate, dipisah koma. Contoh: 0.001,0.0003",
    )
    parser.add_argument(
        "--weight_decay_grid",
        type=str,
        default="0,0.0001",
        help="Grid weight decay, dipisah koma. Contoh: 0,0.0001",
    )
    parser.add_argument(
        "--use_dagshub",
        action="store_true",
        help="Jika diaktifkan, tracking akan diarahkan ke DagsHub (Advanced)",
    )
    parser.add_argument(
        "--dagshub_owner",
        type=str,
        default=os.getenv("DAGSHUB_REPO_OWNER", ""),
        help="Owner repo DagsHub (atau set env DAGSHUB_REPO_OWNER)",
    )
    parser.add_argument(
        "--dagshub_repo",
        type=str,
        default=os.getenv("DAGSHUB_REPO_NAME", ""),
        help="Nama repo DagsHub (atau set env DAGSHUB_REPO_NAME)",
    )
    parser.add_argument(
        "--dagshub_token",
        type=str,
        default=os.getenv("DAGSHUB_TOKEN", ""),
        help="Token DagsHub (atau set env DAGSHUB_TOKEN)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device untuk feature extraction",
    )
    return parser.parse_args()


def resolve_image_path(image_path: str, raw_dir: str) -> str:
    if os.path.exists(image_path):
        return image_path

    raw_dir = os.path.abspath(raw_dir)
    marker = "Vegetables_raw"
    if marker in image_path:
        suffix = image_path.split(marker, 1)[1].lstrip("\\/")
        candidate = os.path.join(raw_dir, suffix)
        if os.path.exists(candidate):
            return candidate

    return image_path


def stratified_limit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if not limit or limit <= 0 or limit >= len(df):
        return df
    n_classes = int(df["label_encoded"].nunique())
    per_class = max(1, limit // max(1, n_classes))
    parts = []
    for _, g in df.groupby("label_encoded"):
        parts.append(g.sample(n=min(len(g), per_class), random_state=42))
    sampled = pd.concat(parts, axis=0)
    remaining = limit - len(sampled)
    if remaining > 0:
        leftover = df.drop(sampled.index, errors="ignore")
        extra = leftover.sample(n=min(remaining, len(leftover)), random_state=42)
        df2 = pd.concat([sampled, extra], ignore_index=False)
    else:
        df2 = sampled
    return df2.sample(frac=1.0, random_state=42).reset_index(drop=True)


def load_split_csv(data_dir: str, filename: str, raw_dir: str, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = Path(data_dir) / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns or "label_encoded" not in df.columns:
        raise ValueError(
            f"Format CSV tidak sesuai. Wajib ada kolom image_path dan label_encoded. Kolom sekarang: {list(df.columns)}"
        )

    df = stratified_limit(df, limit)
    df["image_path"] = df["image_path"].astype(str).map(lambda p: resolve_image_path(p, raw_dir))
    x_paths = df["image_path"].to_numpy()
    y = df["label_encoded"].to_numpy(dtype=np.int64)
    return x_paths, y


@dataclass(frozen=True)
class TuneConfig:
    lr: float
    weight_decay: float


def parse_float_grid(s: str) -> list[float]:
    vals: list[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def maybe_init_dagshub(owner: str, repo: str, token: str) -> None:
    if not owner or not repo:
        hint = ""
        if DOTENV_PATH_LOADED:
            hint = f" (loaded .env: {DOTENV_PATH_LOADED})"
        raise ValueError(
            "DagsHub butuh --dagshub_owner dan --dagshub_repo "
            "(atau env DAGSHUB_REPO_OWNER/DAGSHUB_REPO_NAME)." + hint
        )
    if token:
        os.environ["DAGSHUB_TOKEN"] = token
    import dagshub  # type: ignore

    dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)


def load_label_names(data_dir: str, n_classes: int) -> list[str]:
    """Load class names from label_mapping.json if present."""
    mapping_path = Path(data_dir) / "label_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        try:
            names = [raw[str(i)] for i in range(len(raw))]
        except Exception:
            pairs = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])
            names = [v for _, v in pairs]
        if len(names) >= n_classes:
            return names[:n_classes]
    return [f"class_{i}" for i in range(n_classes)]


def sanitize_metric_key(name: str) -> str:
    safe = name.strip().lower()
    for ch in [" ", "/", "\\", ":", "-", "."]:
        safe = safe.replace(ch, "_")
    return "".join(c for c in safe if (c.isalnum() or c == "_"))


def log_pytorch_model_compat(model: torch.nn.Module, model_name: str = "model") -> None:
    """Avoid MLflow deprecation warning when possible (artifact_path -> name)."""
    try:
        mlflow.pytorch.log_model(model, name=model_name)
    except TypeError:
        mlflow.pytorch.log_model(model, model_name)


def save_confusion_matrix(cm: np.ndarray, out_path: str, class_names: list[str]) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 9},
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_per_class_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_csv: str,
    out_png: str,
) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    rows = []
    for idx, name in enumerate(class_names):
        if name not in report:
            continue
        rows.append(
            {
                "class_id": idx,
                "class_name": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="class_name", y="f1")
    plt.title("Per-class F1 Comparison")
    plt.xlabel("Class")
    plt.ylabel("F1")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: str,
    epoch_idx: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = torch.as_tensor(yb, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == yb).sum().item())
        total += int(xb.size(0))

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    mlflow.log_metric("train_loss", float(avg_loss), step=epoch_idx)
    mlflow.log_metric("train_accuracy", float(acc), step=epoch_idx)
    return float(avg_loss), float(acc)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend([int(v) for v in yb])
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("Vegetable Classification - Skilled")
    print("Transfer Learning: ResNet18 CNN fine-tuning")
    print("Manual MLflow logging (NO autolog)")
    print("=" * 70)

    # Tracking: Local (default) atau DagsHub (Advanced)
    if args.use_dagshub:
        maybe_init_dagshub(args.dagshub_owner, args.dagshub_repo, args.dagshub_token)
        level = "Advanced (4 pts)"
        tracking = "DagsHub"
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        level = "Skilled (3 pts)"
        tracking = "Local"

    mlflow.set_experiment(args.experiment_name)

    print("\n[1] Loading preprocessed CSVs...")
    x_train_paths, y_train = load_split_csv(args.data_dir, "train_ready.csv", args.raw_dir, args.max_train)
    x_test_paths, y_test = load_split_csv(args.data_dir, "test_ready.csv", args.raw_dir, args.max_test)
    n_classes = int(len(np.unique(y_train)))
    class_names = load_label_names(args.data_dir, n_classes)
    print(f"   Train samples: {len(x_train_paths)}")
    print(f"   Test samples:  {len(x_test_paths)}")
    print(f"   Classes:       {n_classes}")

    print("\n[2] Building pretrained CNN (ResNet18)...")
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    device = args.device
    lr_grid = parse_float_grid(args.lr_grid)
    wd_grid = parse_float_grid(args.weight_decay_grid)
    if not lr_grid:
        lr_grid = [1e-3]
    if not wd_grid:
        wd_grid = [0.0]

    tune_grid = [TuneConfig(lr=lr, weight_decay=wd) for lr in lr_grid for wd in wd_grid]

    train_ds = ImagePathDataset(x_train_paths, y_train, transform)
    test_ds = ImagePathDataset(x_test_paths, y_test, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    parent_run_name = f"tuning_resnet18_epochs{args.epochs}_{tracking.lower()}"
    with mlflow.start_run(run_name=parent_run_name):
        mlflow.log_param("level", level)
        mlflow.log_param("tracking", tracking)
        mlflow.log_param("backbone", "resnet18_imagenet")
        mlflow.log_param("device", device)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("max_train", args.max_train)
        mlflow.log_param("max_test", args.max_test)
        mlflow.log_param("train_samples", len(train_ds))
        mlflow.log_param("test_samples", len(test_ds))
        mlflow.log_param("n_classes", n_classes)
        mlflow.log_param("lr_grid", json.dumps(lr_grid))
        mlflow.log_param("weight_decay_grid", json.dumps(wd_grid))

        best_acc = -1.0
        best_cfg: TuneConfig | None = None
        best_run_id: str | None = None

        print("\n[3] Hyperparameter tuning loop...")
        for idx, cfg in enumerate(tune_grid, start=1):
            run_name = f"cfg{idx}_lr{cfg.lr}_wd{cfg.weight_decay}"
            print(f"\n   -> Run {idx}/{len(tune_grid)}: {run_name}")
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_param("lr", cfg.lr)
                mlflow.log_param("weight_decay", cfg.weight_decay)

                # Build model
                model = models.resnet18(weights=weights)
                num_feats = model.fc.in_features
                model.fc = torch.nn.Linear(num_feats, n_classes)
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.fc.parameters():
                    p.requires_grad = True
                model.to(device)

                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(
                    params=[p for p in model.parameters() if p.requires_grad],
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )

                t0 = time.time()
                for ep in range(args.epochs):
                    train_one_epoch(model, train_loader, optimizer, loss_fn, device, ep)
                    y_true, y_pred = evaluate(model, test_loader, device)

                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
                    bal_acc = balanced_accuracy_score(y_true, y_pred)

                    mlflow.log_metric("test_accuracy", float(acc), step=ep)
                    mlflow.log_metric("test_precision_weighted", float(prec), step=ep)
                    mlflow.log_metric("test_recall_weighted", float(rec), step=ep)
                    mlflow.log_metric("test_f1_weighted", float(f1), step=ep)
                    mlflow.log_metric("test_f1_macro", float(f1_macro), step=ep)
                    mlflow.log_metric("test_balanced_accuracy", float(bal_acc), step=ep)

                train_time = time.time() - t0
                mlflow.log_metric("training_time_sec", float(train_time))

                # Final eval + artifacts
                y_true, y_pred = evaluate(model, test_loader, device)
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
                bal_acc = balanced_accuracy_score(y_true, y_pred)

                print(f"      Test accuracy: {acc:.4f}")
                mlflow.log_metric("final_test_accuracy", float(acc))
                mlflow.log_metric("final_test_precision_weighted", float(prec))
                mlflow.log_metric("final_test_recall_weighted", float(rec))
                mlflow.log_metric("final_test_f1_weighted", float(f1))
                mlflow.log_metric("final_test_f1_macro", float(f1_macro))
                mlflow.log_metric("final_test_balanced_accuracy", float(bal_acc))

                # Per-class F1 (tambahan, di luar metric autolog umum)
                per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
                for cls_id, cls_f1 in enumerate(per_class_f1):
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    mlflow.log_metric(f"f1_class_{cls_id}", float(cls_f1))
                    mlflow.log_metric(f"f1_{sanitize_metric_key(cls_name)}", float(cls_f1))

                cm = confusion_matrix(y_true, y_pred)
                safe_name = run_name.replace("/", "_").replace("\\", "_").replace(":", "_")
                cm_path = f"confusion_matrix_{safe_name}.png"
                save_confusion_matrix(cm, cm_path, class_names)
                mlflow.log_artifact(cm_path)

                report_path = f"classification_report_{safe_name}.txt"
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("Classification Report - Vegetable Dataset (CNN TL)\n")
                    f.write("=" * 70 + "\n")
                    f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
                    f.write("\n\nConfig:\n")
                    f.write(json.dumps({"lr": cfg.lr, "weight_decay": cfg.weight_decay, "epochs": args.epochs}, indent=2))
                    f.write("\n")
                mlflow.log_artifact(report_path)

                # Per-class comparison artifacts (with class names)
                per_class_csv = f"per_class_metrics_{safe_name}.csv"
                per_class_png = f"per_class_f1_{safe_name}.png"
                save_per_class_comparison(y_true, y_pred, class_names, per_class_csv, per_class_png)
                mlflow.log_artifact(per_class_csv)
                mlflow.log_artifact(per_class_png)

                # Extra artifacts (≥2 artifacts tambahan selain model)
                run_config_path = f"run_config_{safe_name}.json"
                with open(run_config_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "lr": cfg.lr,
                            "weight_decay": cfg.weight_decay,
                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "device": device,
                        },
                        f,
                        indent=2,
                    )
                mlflow.log_artifact(run_config_path)

                metrics_summary_path = f"metrics_summary_{safe_name}.json"
                with open(metrics_summary_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "final_test_accuracy": float(acc),
                            "final_test_precision_weighted": float(prec),
                            "final_test_recall_weighted": float(rec),
                            "final_test_f1_weighted": float(f1),
                            "final_test_f1_macro": float(f1_macro),
                            "final_test_balanced_accuracy": float(bal_acc),
                        },
                        f,
                        indent=2,
                    )
                mlflow.log_artifact(metrics_summary_path)

                # Model artifact
                log_pytorch_model_compat(model, "model")

                if acc > best_acc:
                    best_acc = float(acc)
                    best_cfg = cfg
                    best_run_id = mlflow.active_run().info.run_id

            # NOTE: After nested run closes, we're back in the parent run.
            # Some UIs (incl. DagsHub) show "No Artifacts" if you're looking at the parent run.
            # To make screenshots easier, also log copies of the artifacts to the parent run.
            try:
                mlflow.log_artifact(cm_path, artifact_path=f"runs/{safe_name}")
                mlflow.log_artifact(report_path, artifact_path=f"runs/{safe_name}")
                mlflow.log_artifact(run_config_path, artifact_path=f"runs/{safe_name}")
                mlflow.log_artifact(metrics_summary_path, artifact_path=f"runs/{safe_name}")
                mlflow.log_artifact(per_class_csv, artifact_path=f"runs/{safe_name}")
                mlflow.log_artifact(per_class_png, artifact_path=f"runs/{safe_name}")
            except Exception:
                pass

        if best_cfg is not None:
            mlflow.log_metric("best_final_test_accuracy", float(best_acc))
            mlflow.log_param("best_lr", best_cfg.lr)
            mlflow.log_param("best_weight_decay", best_cfg.weight_decay)
            if best_run_id:
                mlflow.log_param("best_run_id", best_run_id)

    print("\n" + "=" * 70)
    print("✅ Training selesai.")
    print("Untuk membuka MLflow UI (jalankan dari folder Membangun_model):")
    if args.use_dagshub:
        print("  (Tracking diarahkan ke DagsHub; buka link repo DagsHub kamu)")
    else:
        print("  mlflow ui --backend-store-uri ./mlruns")
        print("Lalu buka: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
