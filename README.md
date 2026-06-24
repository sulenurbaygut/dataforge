# DataForge

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![sklearn](https://img.shields.io/badge/scikit--learn-compatible-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://img.shields.io/badge/tests-passing-brightgreen)

Herhangi bir veri setini içine at — temiz, modele hazır veri al.

Modüler, production-ready veri ön işleme pipeline'ı. Sklearn Pipeline ve GridSearchCV ile tam uyumlu.

---

## Özellikler

| Modül | Ne yapar |
|---|---|
| `DataLoader` | CSV, Excel, JSON, Parquet yükleme |
| `DataProfiler` | Otomatik EDA — korelasyon haritası, dağılımlar, hedef analizi |
| `DataCleaner` | KNN/MICE eksik veri doldurma · IQR/Z-score/Isolation Forest outlier işleme |
| `DataTransformer` | Auto encoding · RobustScaler · interaction/ratio feature engineering |
| `FeatureSelector` | Variance · multicollinearity · Mutual Information filtreleme |
| `DataPipeline` | Tüm adımları config dict ile birleştiren orchestrator |

---

## Kurulum

```bash
git clone https://github.com/sulenurbaygut/dataforge.git
cd dataforge
pip install -r requirements.txt
```

---

## Hızlı Başlangıç

```python
from src.pipeline import DataPipeline

config = {
    "file_path":             "data/raw/train.csv",
    "target_col":            "score",
    "task":                  "regression",
    "missing_strategy":      "knn",
    "outlier_method":        "iqr",
    "outlier_action":        "cap",
    "encoding_method":       "auto",
    "scaling_method":        "robust",
    "run_feature_selection": True,
    "corr_threshold":        0.95,
    "output_path":           "data/processed/clean_data.csv",
}

pipeline = DataPipeline(config)
df_clean = pipeline.run(profile=True)
```

Temiz veri → `data/processed/clean_data.csv`  
EDA grafikleri → `outputs/`

---

## Modüller — Bağımsız Kullanım

### DataCleaner — Method Chaining

```python
from src.cleaner import DataCleaner

df_clean = (
    DataCleaner(df)
    .drop_duplicates()
    .handle_missing(numeric_strategy="knn", exclude_cols=["target"])
    .handle_outliers(method="iqr", action="cap", exclude_cols=["target"])
    .get()
)
```

`numeric_strategy`: `mean` · `median` · `knn` · `iterative`  
`outlier_method`: `iqr` · `zscore` · `isolation_forest`  
`outlier_action`: `cap` · `remove` · `flag`

---

### FeatureSelector — Sklearn Uyumlu

```python
from src.selector import FeatureSelector

selector = (
    FeatureSelector(target_col="score", task="regression")
    .remove_low_variance(threshold=0.01)
    .remove_high_correlation(threshold=0.95)
    .select_by_mutual_info(k=20)
)

selector.fit(X_train, y_train)
X_train_clean = selector.transform(X_train)
X_test_clean  = selector.transform(X_test)
```

### GridSearchCV Entegrasyonu

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("selector", FeatureSelector(target_col="score")),
    ("model",    Ridge())
])

param_grid = {
    "selector__variance_threshold":    [None, 0.01],
    "selector__correlation_threshold": [0.85, 0.90, 0.95],
    "selector__mi_top_k":              [10, 15, 20],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)
```

---

## Proje Yapısı

```
dataforge/
├── src/
│   ├── loader.py        # Çoklu format desteği
│   ├── profiler.py      # Otomatik EDA ve görselleştirme
│   ├── cleaner.py       # Eksik veri + outlier işleme
│   ├── transformer.py   # Encoding, scaling, feature engineering
│   ├── selector.py      # Sklearn-uyumlu özellik seçimi
│   └── pipeline.py      # End-to-end orchestrator
├── tests/
│   └── test_pipeline.py # 16 entegrasyon testi
├── .github/workflows/
│   └── ci.yml           # Python 3.10/3.11/3.12 CI
├── main.py
└── requirements.txt
```

---

## Tasarım Kararları

**Neden RobustScaler varsayılan?**  
StandardScaler mean ve std kullanır — ikisi de outlier'a duyarlı. RobustScaler medyan ve IQR kullanır, outlier sonrası scaling'de daha güvenli.

**Neden korelasyon atarken rastgele değil, hedefe uzaklığa göre?**  
İki sütun yüksek korelasyonluysa birini atmak gerekir. Hangisini atacağın önemli — modele daha az katkı sağlayanı atmak skoru düşürmez.

**Neden `break` korelasyon döngüsünde?**  
`col` elendiğinde diğer partnerleriyle kıyaslamak anlamsız — hem gereksiz hem de hatalı elemelere yol açabilir.

**Neden `reset_index` yok transform'da?**  
Downstream adımlar orijinal index'e bağımlı olabilir. Index sklearn convention'ı gereği korunur.

---

## Testler

```bash
pytest tests/ -v
```

---

## Lisans

MIT
