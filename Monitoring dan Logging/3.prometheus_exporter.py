"""Kriteria 4 (Basic & Skilled) - Prometheus Exporter.

Exporter ini menghasilkan metriks Prometheus berbasis inference, tanpa harus
serve model via HTTP.

Metriks (cukup untuk Basic & Skilled):
- prediction_requests_total (Counter)
- prediction_requests_failed_total (Counter)
- prediction_latency_seconds (Histogram)
- cpu_usage_percent (Gauge)
- memory_usage_bytes (Gauge)

Mode:
- auto (default): pakai LOCAL kalau model.pkl ada, jika tidak fallback HTTP
- local: pakai model.pkl (sklearn Pipeline) dan prediksi lokal
- http: kirim request ke endpoint /invocations

Env vars:
- EXPORTER_PORT=8000
- TRAFFIC_INTERVAL_SECONDS=2
- MODE=auto|local|http
- MODEL_PATH=model.pkl
- MODEL_INVOCATIONS_URL=http://localhost:5000/invocations
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server


PREDICTION_TOTAL = Counter("prediction_requests_total", "Total prediction requests")
PREDICTION_SUCCESS = Counter("prediction_requests_success_total", "Total successful prediction requests")
PREDICTION_FAILED = Counter("prediction_requests_failed_total", "Total failed prediction requests")
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency for prediction requests",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

API_RESPONSE_TIME = Histogram(
    "api_response_time_seconds",
    "End-to-end response time for inference (seconds)",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

PREDICTIONS_BY_LABEL = Counter(
    "model_predictions_by_label_total",
    "Predictions grouped by predicted label",
    ["label"],
)

CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage percentage")
MEMORY_USAGE = Gauge("memory_usage_bytes", "Memory usage in bytes")

REQUESTS_PER_MINUTE = Gauge("requests_per_minute", "Requests per minute")
ERROR_RATE_PERCENT = Gauge("error_rate_percent", "Error rate percentage")
PREDICTION_THROUGHPUT = Gauge("prediction_throughput", "Predictions per second")
ACTIVE_CONNECTIONS = Gauge("active_connections", "Simulated active connections")
MODEL_CONFIDENCE_SCORE = Gauge("model_confidence_score", "Max predicted probability (0..1)")
DATA_DRIFT_SCORE = Gauge("data_drift_score", "Simulated data drift score")


def update_system_metrics() -> None:
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().used)


def generate_random_features(feature_dim: int) -> np.ndarray:
    return np.random.normal(size=(1, feature_dim)).astype(float)


def build_payload(feature_dim: int) -> dict[str, Any]:
    x = generate_random_features(feature_dim).tolist()
    return {"instances": x}


def load_pickle_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def infer_local(model, feature_dim: int) -> None:
    PREDICTION_TOTAL.inc()
    start = time.perf_counter()
    try:
        x = generate_random_features(feature_dim)
        y = model.predict(x)
        PREDICTION_SUCCESS.inc()
        PREDICTIONS_BY_LABEL.labels(label=str(y[0])).inc()

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(x)
                MODEL_CONFIDENCE_SCORE.set(float(np.max(proba)))
            except Exception:
                pass

        elapsed = time.perf_counter() - start
        PREDICTION_LATENCY.observe(elapsed)
        API_RESPONSE_TIME.observe(elapsed)
    except Exception:
        elapsed = time.perf_counter() - start
        PREDICTION_LATENCY.observe(elapsed)
        API_RESPONSE_TIME.observe(elapsed)
        PREDICTION_FAILED.inc()


def infer_http(url: str, feature_dim: int, timeout_s: float = 10.0) -> None:
    import requests

    PREDICTION_TOTAL.inc()
    payload = build_payload(feature_dim)
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        elapsed = time.perf_counter() - start
        PREDICTION_LATENCY.observe(elapsed)
        API_RESPONSE_TIME.observe(elapsed)
        if resp.status_code < 400:
            PREDICTION_SUCCESS.inc()
        if resp.status_code >= 400:
            PREDICTION_FAILED.inc()
    except Exception:
        elapsed = time.perf_counter() - start
        PREDICTION_LATENCY.observe(elapsed)
        API_RESPONSE_TIME.observe(elapsed)
        PREDICTION_FAILED.inc()


def main() -> None:
    metrics_port = int(os.environ.get("EXPORTER_PORT", "8000"))
    interval_s = float(os.environ.get("TRAFFIC_INTERVAL_SECONDS", "2"))
    mode = os.environ.get("MODE", "auto").strip().lower()
    model_path = os.environ.get("MODEL_PATH", "model.pkl")
    model_url = os.environ.get("MODEL_INVOCATIONS_URL", "http://localhost:5000/invocations")

    model = None
    feature_dim = 512

    want_local = False
    if mode in ("auto", "local"):
        if Path(model_path).exists():
            want_local = True
        elif mode == "local":
            raise SystemExit(f"MODE=local tapi file tidak ditemukan: {model_path}")

    if mode == "http":
        want_local = False

    print("=" * 70)
    print("PROMETHEUS EXPORTER (Kriteria 4 - Basic/Skilled)")
    print("=" * 70)
    print(f"Exporter metrics : http://localhost:{metrics_port}/metrics")
    print(f"Interval         : {interval_s}s")
    print(f"Mode             : {mode}")
    if want_local:
        print(f"Inference        : LOCAL (model={model_path}, feature_dim=auto)")
    else:
        print(f"Inference        : HTTP (url={model_url}, feature_dim={feature_dim})")
    print("=" * 70)

    start_http_server(metrics_port)

    # rolling counters for derived gauges
    window_start = time.time()
    window_requests = 0

    while True:
        update_system_metrics()

        # simulated "system/app" gauges
        ACTIVE_CONNECTIONS.set(float(np.random.randint(1, 25)))
        DATA_DRIFT_SCORE.set(float(np.random.uniform(0.0, 1.0)))

        if want_local and model is None:
            try:
                model = load_pickle_model(model_path)
                if hasattr(model, "n_features_in_"):
                    try:
                        feature_dim = int(model.n_features_in_)
                    except Exception:
                        pass
                print(f"✓ Local model loaded: {model_path} (feature_dim={feature_dim})")
            except KeyboardInterrupt:
                print("! Model load interrupted; falling back to HTTP mode")
                want_local = False
            except Exception as e:
                print(f"! Failed to load local model ({e}); falling back to HTTP mode")
                want_local = False

        if want_local and model is not None:
            infer_local(model, feature_dim)
        else:
            infer_http(model_url, feature_dim)

        window_requests += 1
        now = time.time()
        elapsed_window = now - window_start
        if elapsed_window >= 60:
            REQUESTS_PER_MINUTE.set(float(window_requests))
            window_start = now
            window_requests = 0

        # derived rates (best-effort)
        try:
            total = float(PREDICTION_TOTAL._value.get())
            failed = float(PREDICTION_FAILED._value.get())
            ERROR_RATE_PERCENT.set(0.0 if total <= 0 else (failed / total) * 100.0)
        except Exception:
            pass

        PREDICTION_THROUGHPUT.set(0.0 if interval_s <= 0 else (1.0 / interval_s))
        time.sleep(interval_s)


if __name__ == "__main__":
    main()
