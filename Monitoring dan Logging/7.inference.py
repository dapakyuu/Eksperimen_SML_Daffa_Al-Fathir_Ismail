"""Kriteria 4 - Inference client (optional).

Script ini bisa dipakai 2 mode:
1) local  : load model dari file `model.pkl` (sklearn Pipeline) lalu prediksi lokal.
2) http   : kirim request ke endpoint model (mis. MLflow pyfunc REST /invocations).

Default:
- Jika `model.pkl` ada, mode default = local.
- Kalau tidak ada, mode default = http.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference client (local pickle or HTTP)")
    parser.add_argument(
        "--mode",
        choices=["auto", "local", "http"],
        default="auto",
        help="auto=pakai local jika model.pkl ada, kalau tidak pakai http",
    )
    parser.add_argument("--model_path", type=str, default="model.pkl", help="Path ke model pickle")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("MODEL_INVOCATIONS_URL", "http://localhost:5000/invocations"),
        help="Endpoint HTTP untuk mode=http",
    )
    parser.add_argument("--n_requests", type=int, default=int(os.environ.get("N_REQUESTS", "30")))
    parser.add_argument("--delay", type=float, default=float(os.environ.get("DELAY_SECONDS", "0.5")))
    parser.add_argument("--feature_dim", type=int, default=512)
    return parser.parse_args()


def load_pickle_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_local(model, x: np.ndarray):
    # sklearn Pipeline/Estimator supports predict
    return model.predict(x)


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    mode = args.mode
    if mode == "auto":
        mode = "local" if model_path.exists() else "http"

    if mode == "local":
        if not model_path.exists():
            raise SystemExit(f"model_path tidak ditemukan: {model_path}")
        model = load_pickle_model(str(model_path))

        # Tentukan feature_dim dari model jika tersedia
        feature_dim = args.feature_dim
        if hasattr(model, "n_features_in_"):
            try:
                feature_dim = int(getattr(model, "n_features_in_"))
            except Exception:
                pass
        print(f"Mode: local | model={model_path} | feature_dim={feature_dim}")

        for i in range(1, args.n_requests + 1):
            x = np.random.normal(size=(1, feature_dim)).astype(float)
            y = infer_local(model, x)
            print(f"[{i}/{args.n_requests}] pred={y[0]}")
            time.sleep(args.delay)
        return

    # mode == http
    import requests

    print(f"Mode: http | endpoint={args.endpoint}")
    for i in range(1, args.n_requests + 1):
        payload = {"instances": np.random.normal(size=(1, args.feature_dim)).astype(float).tolist()}
        resp = requests.post(args.endpoint, json=payload, timeout=10)
        print(f"[{i}/{args.n_requests}] status={resp.status_code}")
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
