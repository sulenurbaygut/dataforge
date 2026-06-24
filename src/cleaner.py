import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


class DataCleaner:
    """
    Method chaining API ile çalışan veri temizleme motoru.

    Kullanım:
        df_clean = (
            DataCleaner(df)
            .drop_duplicates()
            .handle_missing(numeric_strategy='knn')
            .handle_outliers(method='iqr', action='cap')
            .get()
        )
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._refresh_col_types()

    def _refresh_col_types(self):
        self._numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    # ------------------------------------------------------------------
    # 1. Duplikasyonlar
    # ------------------------------------------------------------------

    def drop_duplicates(self) -> "DataCleaner":
        """Tamamen aynı olan satırları kaldırır."""
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        if removed:
            print(f"✅ Duplikasyon temizlendi | {removed} satır kaldırıldı")
        else:
            print("ℹ️  Duplikasyon bulunamadı.")
        return self

    # ------------------------------------------------------------------
    # 2. Eksik Değerler
    # ------------------------------------------------------------------

    def handle_missing(
        self,
        numeric_strategy: str = "knn",
        categorical_strategy: str = "mode",
        knn_neighbors: int = 5,
        constant_value: str = "Unknown",
        exclude_cols: list = None,
    ) -> "DataCleaner":
        """
        Eksik değerleri seçilen stratejiye göre doldurur.

        numeric_strategy:
            'mean'      – Ortalama (hızlı, basit)
            'median'    – Medyan (aykırı değere dayanıklı)
            'knn'       – K-NN Imputer (dengeli doğruluk/hız) ✅ varsayılan
            'iterative' – MICE / Çok değişkenli (en doğru, yavaş)

        categorical_strategy:
            'mode'     – En sık görülen değer ✅
            'constant' – Sabit değer (constant_value parametresi)
        """
        exclude_cols = exclude_cols or []
        num_cols = [c for c in self._numeric_cols if c not in exclude_cols]
        cat_cols = [c for c in self._categorical_cols if c not in exclude_cols]

        missing_before = self.df.isnull().sum().sum()
        if missing_before == 0:
            print("ℹ️  Eksik değer yok, imputation atlandı.")
            return self

        # ── Sayısal ──────────────────────────────────────────────────
        num_missing = self.df[num_cols].isnull().sum().sum() if num_cols else 0
        if num_missing > 0:
            if numeric_strategy == "knn":
                imp = KNNImputer(n_neighbors=knn_neighbors)
            elif numeric_strategy == "iterative":
                imp = IterativeImputer(random_state=42, max_iter=10)
            elif numeric_strategy in ("mean", "median"):
                imp = SimpleImputer(strategy=numeric_strategy)
            else:
                raise ValueError(f"Geçersiz numeric_strategy: '{numeric_strategy}'")

            self.df[num_cols] = imp.fit_transform(self.df[num_cols])

        # ── Kategorik ────────────────────────────────────────────────
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                fill = (
                    self.df[col].mode()[0]
                    if categorical_strategy == "mode"
                    else constant_value
                )
                self.df[col] = self.df[col].fillna(fill)

        missing_after = self.df.isnull().sum().sum()
        print(
            f"✅ Eksik değer işlendi | "
            f"Strateji: {numeric_strategy} | "
            f"{missing_before} → {missing_after}"
        )
        return self

    # ------------------------------------------------------------------
    # 3. Aykırı Değerler
    # ------------------------------------------------------------------

    def handle_outliers(
        self,
        method: str = "iqr",
        action: str = "cap",
        threshold: float = None,
        columns: list = None,
        exclude_cols: list = None,
        contamination: float = 0.05,
    ) -> "DataCleaner":
        """
        Aykırı değerleri tespit eder ve işler.

        method:
            'iqr'              – IQR tabanlı (genel amaçlı) ✅ varsayılan
            'zscore'           – Z-score (normal dağılım varsayımı)
            'isolation_forest' – ML tabanlı (çok boyutlu, en güçlü)

        action:
            'cap'    – Sınır değere baskıla (veri kaybı yok) ✅
            'remove' – Satırı sil
            'flag'   – Yeni binary sütunla işaretle, silme

        threshold:
            IQR için varsayılan 1.5, Z-score için 3.0
        """
        exclude_cols = exclude_cols or []
        target_cols = columns or [
            c for c in self._numeric_cols if c not in exclude_cols
        ]

        # ── Isolation Forest ─────────────────────────────────────────
        if method == "isolation_forest":
            data = self.df[target_cols].fillna(self.df[target_cols].median())
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(data)
            mask = preds == -1
            n = int(mask.sum())

            if action == "remove":
                self.df = self.df[~mask].reset_index(drop=True)
                print(f"✅ Isolation Forest | {n} satır silindi")
            elif action == "cap":
                # Isolation Forest'te cap uygulanamaz, flag'e düşülür
                self.df["is_outlier"] = mask.astype(int)
                print(f"✅ Isolation Forest | {n} aykırı değer flaglendi (cap yok)")
            elif action == "flag":
                self.df["is_outlier"] = mask.astype(int)
                print(f"✅ Isolation Forest | {n} aykırı değer flaglendi")
            return self

        # ── IQR / Z-score ────────────────────────────────────────────
        default_threshold = 1.5 if method == "iqr" else 3.0
        thr = threshold if threshold is not None else default_threshold

        total_affected = 0
        removed_mask = pd.Series([False] * len(self.df), index=self.df.index)

        for col in target_cols:
            if method == "iqr":
                Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lo, hi = Q1 - thr * IQR, Q3 + thr * IQR
            else:  # zscore
                mu, sigma = self.df[col].mean(), self.df[col].std()
                lo, hi = mu - thr * sigma, mu + thr * sigma

            outlier_mask = (self.df[col] < lo) | (self.df[col] > hi)
            n = int(outlier_mask.sum())

            if n == 0:
                continue

            if action == "cap":
                self.df[col] = np.clip(self.df[col], lo, hi)
                total_affected += n
            elif action == "remove":
                removed_mask |= outlier_mask
            elif action == "flag":
                self.df[f"{col}_outlier"] = outlier_mask.astype(int)
                total_affected += n

        if action == "remove":
            before = len(self.df)
            self.df = self.df[~removed_mask].reset_index(drop=True)
            print(
                f"✅ Outlier ({method.upper()}) | "
                f"{before - len(self.df)} satır silindi"
            )
        else:
            print(
                f"✅ Outlier ({method.upper()}) | "
                f"{total_affected} değer {action} edildi"
            )

        self._refresh_col_types()
        return self

    # ------------------------------------------------------------------
    # 4. Çıktı
    # ------------------------------------------------------------------

    def get(self) -> pd.DataFrame:
        return self.df.reset_index(drop=True)
