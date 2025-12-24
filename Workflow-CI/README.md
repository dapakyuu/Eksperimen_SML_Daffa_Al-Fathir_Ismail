# Workflow CI - MLflow Project with GitHub Actions

**Kriteria 3: Advanced Level (4 pts)**

Proyek ini mengimplementasikan Workflow CI untuk automated model training menggunakan:
- ✅ MLflow Project
- ✅ GitHub Actions CI
- ✅ Artifact upload ke GitHub Actions

> Catatan: Sesuai kriteria 3 advanced, bagian Docker dapat ditambahkan (mlflow build-docker + push ke Docker Hub).

---

## 📁 Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml          # GitHub Actions workflow
├── MLProject/
│   ├── MLproject                  # MLflow Project config
│   ├── conda.yaml                 # Conda environment (optional)
│   ├── python_env.yaml            # Python environment
│   ├── modelling.py               # Training script
│   └── Vegetables_preprocessing/  # Preprocessed dataset
│       ├── train_ready.csv
│       ├── test_ready.csv
│       ├── val_ready.csv
│       ├── label_mapping.json
│       └── preprocessing_summary.json
└── README.md
```

---

## 🚀 Cara Menjalankan

### Local

```bash
cd Workflow-CI/MLProject
mlflow run . --env-manager=local
```

### GitHub Actions

Workflow berjalan saat:
- push ke `main/master` yang mengubah `MLProject/**`
- Pull Request
- manual trigger `workflow_dispatch`

---

## 📦 Output

Setiap run akan menghasilkan:
- MLflow tracking folder `MLProject/mlruns/`
- artifacts:
  - `confusion_matrix.png`
  - `classification_report.txt`
  - model di MLflow artifacts (`model/`)

---
