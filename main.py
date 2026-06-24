"""
Data Preprocessing Pipeline — Giriş Noktası
============================================
Kullanım:
    python main.py

Config dict'ini düzenleyerek herhangi bir veri setine uyarlayabilirsiniz.
"""

from src.pipeline import DataPipeline

config = {
    # ── Veri ──────────────────────────────────────────────────────────
    "file_path":             "data/raw/train.csv",   # CSV | Excel | JSON | Parquet
    "target_col":            "score",                # Hedef sütun adı
    "task":                  "regression",           # 'regression' | 'classification'

    # ── Temizlik ──────────────────────────────────────────────────────
    "missing_strategy":      "knn",       # 'mean' | 'median' | 'knn' | 'iterative'
    "outlier_method":        "iqr",       # 'iqr' | 'zscore' | 'isolation_forest'
    "outlier_action":        "cap",       # 'cap' | 'remove' | 'flag'

    # ── Dönüşüm ───────────────────────────────────────────────────────
    "encoding_method":       "auto",      # 'auto' | 'onehot' | 'label' | 'target'
    "scaling_method":        "robust",    # 'standard' | 'minmax' | 'robust'
    "exclude_from_scaling":  [],          # Örn: ['id', 'student_id']
    "exclude_from_encoding": [],

    # Özellik mühendisliği (opsiyonel, boş bırakılabilir)
    "interaction_pairs":     [],          # Örn: [('study_hours', 'attendance')]
    "ratio_pairs":           [],          # Örn: [('correct', 'total_questions')]

    # ── Özellik Seçimi ────────────────────────────────────────────────
    "run_feature_selection": True,
    "corr_threshold":        0.95,        # Bu eşiğin üzerindeki korelasyonlar atılır
    "mi_top_k":              None,        # Kaç özellik kalsın? None → hepsini tut

    # ── Çıktı ─────────────────────────────────────────────────────────
    "output_path":           "data/processed/clean_data.csv",
}

if __name__ == "__main__":
    pipeline = DataPipeline(config)
    df_clean = pipeline.run(profile=True)

    print(f"Model eğitimine hazır veri — ilk 5 satır:")
    print(df_clean.head())
