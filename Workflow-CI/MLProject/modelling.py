"""Kriteria 3 - MLflow Project (Vegetables).

Script ini dibuat agar mirip dengan Kriteria 2 Basic modelling:
- Transfer learning: pretrained ResNet18 (feature extractor dibekukan)
- Classifier: Scikit-Learn (LogisticRegression)
- Logging: MLflow autolog + beberapa metric + artifacts

Catatan penting untuk CI:
- Workflow akan mengunduh dataset (Vegetables_raw) dan menjalankan preprocessing,
  sehingga CSV berisi path yang valid di runner.
"""

from __future__ import annotations

import argparse
import json
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


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


def resolve_here(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Vegetables classifier (transfer learning + sklearn) with MLflow autolog"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(resolve_here("Vegetables_preprocessing")),
        help="Folder berisi train_ready.csv/test_ready.csv",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Vegetables - CI (MLflow Project)",
        help="Nama MLflow experiment",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--max_train",
        type=int,
        default=200,
        help="Default 200 sesuai modelling sebelumnya (0=semua)",
    )
    parser.add_argument(
        "--max_test",
        type=int,
        default=100,
        help="Default 100 sesuai modelling sebelumnya (0=semua)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()

def load_split_csv(data_dir: str, filename: str, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = Path(data_dir) / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns or "label_encoded" not in df.columns:
        raise ValueError(
            f"Format CSV tidak sesuai. Wajib ada kolom image_path dan label_encoded. Kolom sekarang: {list(df.columns)}"
        )

    if limit and limit > 0:
        n_total = len(df)
        if limit < n_total:
            n_classes = int(df["label_encoded"].nunique())
            per_class = max(1, limit // max(1, n_classes))
            parts = []
            for _, g in df.groupby("label_encoded"):
                parts.append(g.sample(n=min(len(g), per_class), random_state=42))
            sampled = pd.concat(parts, axis=0)

            if len(sampled) > limit:
                sampled = sampled.sample(n=limit, random_state=42)

            df = sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)

    x_paths = df["image_path"].astype(str).to_numpy()
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


def load_label_mapping(data_dir: str) -> list[str] | None:
    mapping_path = Path(data_dir) / "label_mapping.json"
    if not mapping_path.exists():
        return None
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping = {int(k): str(v) for k, v in data.items()}
    return [mapping[i] for i in range(len(mapping))]


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("Vegetables Classification - MLflow Project")
    print("Transfer Learning: ResNet18 feature extractor + LogisticRegression")
    print("=" * 70)

    mlflow.sklearn.autolog()

    def train_and_log() -> None:
        mlflow.log_param("feature_extractor", "resnet18_imagenet")
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("max_train", args.max_train)
        mlflow.log_param("max_test", args.max_test)
        mlflow.log_param("device", args.device)

        print("\n[1] Loading preprocessed CSVs...")
        x_train_paths, y_train = load_split_csv(args.data_dir, "train_ready.csv", args.max_train)
        x_test_paths, y_test = load_split_csv(args.data_dir, "test_ready.csv", args.max_test)
        class_names = load_label_mapping(args.data_dir)

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
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision_weighted", precision)
        mlflow.log_metric("test_recall_weighted", recall)
        mlflow.log_metric("test_f1_weighted", f1)

        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

        print("\n[6] Artifacts...")
        labels = None
        if class_names is not None:
            labels = list(range(len(class_names)))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        heatmap_kwargs = {"annot": True, "fmt": "d", "cmap": "Blues"}
        if class_names is not None:
            heatmap_kwargs["xticklabels"] = class_names
            heatmap_kwargs["yticklabels"] = class_names
        sns.heatmap(cm, **heatmap_kwargs)
        plt.title("Confusion Matrix - Vegetables")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_path = resolve_here("confusion_matrix.png")
        plt.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(cm_path))

        report_kwargs = {"zero_division": 0}
        if class_names is not None:
            report_kwargs["labels"] = labels
            report_kwargs["target_names"] = class_names
        report = classification_report(y_test, y_pred, **report_kwargs)
        report_path = resolve_here("classification_report.txt")
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        # Fix warning: use `name=` instead of deprecated `artifact_path`
        mlflow.sklearn.log_model(clf, name="model")

    active = mlflow.active_run()
    env_run_id = os.environ.get("MLFLOW_RUN_ID")

    # When executed via `mlflow run`, MLflow injects MLFLOW_RUN_ID/MLFLOW_EXPERIMENT_ID.
    # We must attach to that run and MUST NOT call set_experiment(), otherwise MLflow
    # raises an experiment-id mismatch error.
    if active is None and env_run_id:
        with mlflow.start_run(run_id=env_run_id) as run:
            train_and_log()
            print(f"\n🔑 MLFLOW_RUN_ID={run.info.run_id}")
    elif active is None:
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run() as run:
            train_and_log()
            print(f"\n🔑 MLFLOW_RUN_ID={run.info.run_id}")
    else:
        train_and_log()
        run = mlflow.active_run()
        if run:
            print(f"\n🔑 MLFLOW_RUN_ID={run.info.run_id}")

    print("\n✅ Training selesai.")


if __name__ == "__main__":
    main()
