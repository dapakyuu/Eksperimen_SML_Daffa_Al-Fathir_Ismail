# Monitoring dan Logging - Vegetables (Kriteria 4 - Basic)

Target **Basic (2 pts)**:
- Serving model di local environment
- Monitoring dengan Prometheus minimal **3 metriks**
- Visualisasi di Grafana dengan metriks yang sama

Folder bukti (isi dengan screenshot):
- `1.bukti_serving/`
- `4.bukti_monitoring_Prometheus/`
- `5.bukti_monitoring_Grafana/`

---

## 0) Prasyarat

- Docker Desktop (untuk Prometheus + Grafana)
- Python 3.10+ (untuk exporter)

---

## 1) Jalankan Model Serving (wajib untuk bukti serving)

Karena di Kriteria 3 kamu sudah push Docker image ke Docker Hub, cara paling simpel:

1. Pull image (ganti username sesuai Docker Hub kamu):

```bash
docker pull <DOCKERHUB_USERNAME>/vegetables-mlflow:latest
```

2. Run container dan expose ke localhost port 5000:

```bash
docker run --rm -p 5000:8080 <DOCKERHUB_USERNAME>/vegetables-mlflow:latest
```

Catatan:
- Banyak image hasil `mlflow models build-docker` listen di port container **8080**.
- Endpoint MLflow serving biasanya `POST /invocations`.

📸 Screenshot bukti:
- Container running + port mapping (Terminal / Docker Desktop) simpan di `1.bukti_serving/`.

---

## 2) Jalankan Exporter (Prometheus metrics)

Terminal 1:

```bash
cd "Monitoring dan Logging"
pip install -r requirements.txt
python 3.prometheus_exporter.py
```

Cek metrics di browser:
- http://localhost:8000/metrics

Metriks utama (Basic):
- `prediction_requests_total`
- `prediction_requests_failed_total`
- `prediction_latency_seconds`

---

## 3) Jalankan Prometheus & Grafana

Terminal 2:

```bash
cd "Monitoring dan Logging"
docker-compose up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / admin123)

---

## 4) Bukti Prometheus (minimal 3 metriks)

1. Buka Prometheus → **Status → Targets** pastikan job `vegetables-exporter` status **UP**.
   - Screenshot → `4.bukti_monitoring_Prometheus/`

2. Di tab **Graph**, jalankan query dan screenshot minimal 3:
   - `prediction_requests_total`
   - `prediction_latency_seconds`
   - `cpu_usage_percent`

---

## 5) Bukti Grafana (metriks sama dengan Prometheus)

1. Grafana → **Add data source** → Prometheus
   - URL: `http://prometheus:9090`

2. Buat dashboard (nama dashboard = **username Dicoding** kamu)

3. Buat minimal 3 panel dan screenshot:
   - Total requests: `prediction_requests_total`
   - Latency: `prediction_latency_seconds`
   - CPU: `cpu_usage_percent`

Simpan ke `5.bukti_monitoring_Grafana/`.

---

## (Optional) Generate traffic manual

Kalau mau menambah traffic secara manual:

```bash
python 7.inference.py
```

Exporter default sudah generate traffic otomatis setiap 2 detik.
