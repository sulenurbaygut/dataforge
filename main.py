"""
DataForge — Giriş Noktası
Kullanım: python main.py
"""
from sklearn.model_selection import train_test_split
from src.pipeline import DataPipeline
from src.trainer import ModelTrainer

# ── 1. Veri Ön İşleme ────────────────────────────────────────────────
config = {
    "file_path":             "data/raw/train.csv",
    "target_col":            "score",
    "task":                  "regression",
    "missing_strategy":      "knn",
    "outlier_method":        "iqr",
    "outlier_action":        "cap",
    "encoding_method":       "auto",
    "scaling_method":        "robust",
    "exclude_from_scaling":  [],
    "exclude_from_encoding": [],
    "interaction_pairs":     [],
    "ratio_pairs":           [],
    "run_feature_selection": True,
    "corr_threshold":        0.95,
    "mi_top_k":              None,
    "output_path":           "data/processed/clean_data.csv",
}

pipeline = DataPipeline(config)
df_clean = pipeline.run(profile=True)

# ── 2. Train/Test Ayır ───────────────────────────────────────────────
target = config["target_col"]
X = df_clean.drop(columns=[target])
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Model Karşılaştırma ───────────────────────────────────────────
trainer = ModelTrainer(task=config["task"], cv=5)
results = trainer.compare_models(X_train, y_train)

# ── 4. Hyperparameter Tuning (en iyi model otomatik seçilir) ─────────
best = trainer.tune(X_train, y_train, n_trials=50)

# ── 5. Ensemble ──────────────────────────────────────────────────────
ensemble = trainer.build_ensemble(X_train, y_train, top_n=3, method="stacking")

# ── 6. Test Seti Değerlendirme ────────────────────────────────────────
print("\n── Tuned Model ──")
trainer.evaluate(best, X_test, y_test)

print("\n── Ensemble ──")
trainer.evaluate(ensemble, X_test, y_test)
