""" 
Kriteria 2 - Basic (2 pts)

- Melatih model machine learning (Scikit-Learn) menggunakan MLflow Tracking UI (lokal)
- Menggunakan autolog dari MLflow pada file modelling.py
- Tanpa hyperparameter tuning

Catatan implementasi:
- Agar training cepat dan tetap "transfer learning", script ini memakai model pretrained
  (ResNet18) sebagai feature extractor (dibekukan), lalu melatih classifier Scikit-Learn
  (LogisticRegression) di atas fitur tersebut.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Iterable[str], transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Vegetable classifier (transfer learning + sklearn) with MLflow autolog"
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
        default="Vegetable Classification Basic (Transfer Learning)",
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device untuk feature extraction",
    )
    return parser.parse_args()


def resolve_image_path(image_path: str, raw_dir: str) -> str:
    """Resolve path dari CSV.

    CSV hasil preprocessing bisa berisi absolute path lokal. Jika path tidak ada,
    kita coba remap berdasarkan folder Vegetables_raw.
    """
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


def load_split_csv(data_dir: str, filename: str, raw_dir: str, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = Path(data_dir) / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns or "label_encoded" not in df.columns:
        raise ValueError(
            f"Format CSV tidak sesuai. Wajib ada kolom image_path dan label_encoded. Kolom sekarang: {list(df.columns)}"
        )

    if limit and limit > 0:
        # Ambil sampel secara stratified agar tetap mencakup banyak kelas.
        n_total = len(df)
        if limit < n_total:
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
                df = pd.concat([sampled, extra], ignore_index=False)
            else:
                df = sampled
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df["image_path"] = df["image_path"].astype(str).map(lambda p: resolve_image_path(p, raw_dir))
    x_paths = df["image_path"].to_numpy()
    y = df["label_encoded"].to_numpy(dtype=np.int64)
    return x_paths, y


@torch.inference_mode()
def extract_features(
    model: torch.nn.Module,
    transform,
    image_paths: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    dataset = ImagePathDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feats: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        feats.append(out.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("Vegetable Classification - Basic MLflow Training")
    print("Transfer Learning: ResNet18 feature extractor + LogisticRegression")
    print("=" * 70)

    # MLflow local tracking (file store) agar tidak bergantung server yang sedang berjalan
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment_name)
    mlflow.sklearn.autolog()

    print("\n[1] Loading preprocessed CSVs...")
    x_train_paths, y_train = load_split_csv(args.data_dir, "train_ready.csv", args.raw_dir, args.max_train)
    x_test_paths, y_test = load_split_csv(args.data_dir, "test_ready.csv", args.raw_dir, args.max_test)

    print(f"   Train samples: {len(x_train_paths)}")
    print(f"   Test samples:  {len(x_test_paths)}")

    print("\n[2] Building pretrained feature extractor (ResNet18)...")
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    backbone = models.resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    device = args.device
    backbone.to(device)

    with mlflow.start_run():
        # Log a few helpful params (autolog will also record sklearn params)
        mlflow.log_param("feature_extractor", "resnet18_imagenet")
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("max_train", args.max_train)
        mlflow.log_param("max_test", args.max_test)
        mlflow.log_param("device", device)

        print("\n[3] Extracting features...")
        x_train = extract_features(backbone, transform, x_train_paths, args.batch_size, device)
        x_test = extract_features(backbone, transform, x_test_paths, args.batch_size, device)
        print(f"   X_train shape: {x_train.shape}")
        print(f"   X_test shape:  {x_test.shape}")

        print("\n[4] Training Scikit-Learn classifier (no tuning)...")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train, y_train)

        print("\n[5] Evaluating...")
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Ensure metrics appear clearly in MLflow UI (autolog may not log these by default)
        mlflow.log_metric("test_accuracy", float(accuracy))
        mlflow.log_metric("test_precision_weighted", float(precision))
        mlflow.log_metric("test_recall_weighted", float(recall))
        mlflow.log_metric("test_f1_weighted", float(f1))

        print("\n[6] Model Performance:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

        report = classification_report(y_test, y_pred, zero_division=0)
        report_path = "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Classification Report - Vegetable Dataset\n")
            f.write("=" * 70 + "\n")
            f.write(report)
            f.write("\n")

        mlflow.log_artifact(report_path)
        print(f"\n[7] Saved artifact: {report_path}")

    print("\n" + "=" * 70)
    print("✅ Training selesai.")
    print("Untuk membuka MLflow UI (jalankan dari folder Membangun_model):")
    print("  mlflow ui --backend-store-uri ./mlruns")
    print("Lalu buka: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
