# MSML Final Project — Klasifikasi Gambar Sayuran (Vegetables)

Repo ini berisi pipeline end-to-end untuk **preprocessing dataset gambar sayuran**, **training & tracking eksperimen dengan MLflow**, **workflow CI (MLflow Project + GitHub Actions)**, serta **serving + monitoring (Prometheus & Grafana)**.

## Ringkasan

- **Dataset**: struktur folder `Vegetables_raw/{train,test,validation}/<class>/*.jpg`.
- **Preprocessing**: menghasilkan CSV metadata (`*_ready.csv`) + mapping label.
- **Training (Kriteria 2)**:
  - Basic: **ResNet18 sebagai feature extractor** + **Scikit-Learn LogisticRegression** (MLflow autolog).
  - Skilled/Advanced: **CNN transfer learning ResNet18** + **hyperparameter tuning** (manual logging), opsional tracking ke **DagsHub**.
- **Workflow CI (Kriteria 3)**: MLflow Project yang bisa dijalankan lokal maupun via GitHub Actions.
- **Monitoring & Logging (Kriteria 4)**: exporter Prometheus + dashboard Grafana.

## Struktur Folder (high-level)

- `Eksperimen_MSML_Daffa_Al-Fathir_Ismail/`
  - `Vegetables_raw/` — dataset gambar (train/test/validation)
  - `preprocessing/automate_Daffa.py` — script preprocessing
- `Membangun_model/`
  - `modelling.py` — training Basic (autolog)
  - `modelling_tuning.py` — training Skilled/Advanced (tuning + manual logging)
  - `Vegetables_preprocessing/` — output preprocessing (CSV + mapping)
  - `mlruns/` — MLflow tracking (file store)
- `Workflow-CI/`
  - MLflow Project + workflow GitHub Actions
- `Monitoring dan Logging/`
  - `docker-compose.yml`, Prometheus config, exporter, dan bukti screenshot

## Prasyarat

- Python **3.10+**
- (Opsional) GPU/CUDA untuk ekstraksi fitur lebih cepat
- Docker Desktop (untuk Prometheus + Grafana)

## Setup Environment

Disarankan pakai virtualenv.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependensi training:

```bash
pip install -r "Membangun_model/requirements.txt"
```

Install dependensi monitoring (jika perlu menjalankan exporter):

```bash
pip install -r "Monitoring dan Logging/requirements.txt"
```

## 1) Preprocessing Dataset

Script preprocessing akan membuat file:
- `train_ready.csv`, `test_ready.csv`, `val_ready.csv`
- `label_mapping.json`
- `preprocessing_summary.json` (opsional)

Contoh menjalankan preprocessing (output diarahkan ke folder training):

```bash
cd "Eksperimen_MSML_Daffa_Al-Fathir_Ismail/preprocessing"
python automate_Daffa.py --dataset_dir "../Vegetables_raw" --output_dir "../../Membangun_model/Vegetables_preprocessing"
```

Jika CSV sudah tersedia, langkah ini bisa dilewati.

## 2) Training + MLflow Tracking (Kriteria 2)

### A. Basic (autolog) — `Membangun_model/modelling.py`

Jalankan training dari folder `Membangun_model` agar default path sesuai:

```bash
cd "Membangun_model"
python modelling.py
```

Opsional (supaya cepat) batasi jumlah data:

```bash
python modelling.py --max_train 500 --max_test 200
```

Buka MLflow UI (tracking disimpan di `Membangun_model/mlruns`):

```bash
mlflow ui --backend-store-uri ./mlruns
```

Lalu akses: http://localhost:5000

### B. Skilled/Advanced (tuning + manual logging) — `Membangun_model/modelling_tuning.py`

```bash
cd "Membangun_model"
python modelling_tuning.py --epochs 2 --lr_grid "0.001,0.0003" --weight_decay_grid "0,0.0001"
```

Opsional: tracking ke DagsHub (Advanced). Siapkan env var atau file `.env` yang berisi:
- `DAGSHUB_REPO_OWNER`
- `DAGSHUB_REPO_NAME`
- `DAGSHUB_TOKEN`

Lalu jalankan:

```bash
python modelling_tuning.py --use_dagshub --dagshub_owner "<owner>" --dagshub_repo "<repo>" --dagshub_token "<token>"
```

## 3) Workflow CI (Kriteria 3) — MLflow Project

Untuk detail lengkap, lihat README di folder `Workflow-CI/`.

Menjalankan MLflow Project secara lokal:

```bash
cd "Workflow-CI/MLProject"
mlflow run . --env-manager=local
```

Output eksperimen akan tersimpan di `Workflow-CI/MLProject/mlruns/`.

## 4) Serving + Monitoring (Kriteria 4)

Untuk langkah detail, lihat `Monitoring dan Logging/README.md`.

Ringkasnya:

1) (Opsional) Jalankan model serving (mis. image Docker MLflow) di port lokal.

2) Jalankan Prometheus exporter:

```bash
cd "Monitoring dan Logging"
python 3.prometheus_exporter.py
```

Cek metrics: http://localhost:8000/metrics

3) Jalankan Prometheus & Grafana via Docker:

```bash
cd "Monitoring dan Logging"
docker-compose up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default: `admin` / `admin123`)

## Catatan Reproducibility

- File CSV preprocessing menyimpan `image_path`. Jika path asal adalah absolute path lokal, script training akan mencoba **fallback** dengan cara me-remap ke folder `Vegetables_raw`.
- Untuk penggunaan lintas mesin/OS, disarankan menjalankan preprocessing ulang agar path di CSV sesuai environment saat ini.

## Lisensi & Kredit

- Dataset dan model pretrained mengikuti lisensi sumbernya masing-masing.

---

Jika kamu mau, saya bisa sekalian rapikan README ini agar cocok untuk tampilan GitHub (mis. tambah badge, contoh request inference, dan penjelasan output artifacts) tanpa mengubah scope UX/fitur proyek.