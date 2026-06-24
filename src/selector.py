import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Sklearn Pipeline ve GridSearchCV ile tam uyumlu özellik seçim motoru.

    Neden önceki versiyondan farklı:
    ─────────────────────────────────
    • Tüm konfigürasyon __init__ içinde → sklearn.clone() ve GridSearchCV çalışır
    • Method chaining hâlâ destekleniyor (parametreleri günceller)
    • features_to_drop_ set olarak tutulur → duplicate drop hatası yok
    • mi_scores_ fit sonrası inspect edilebilir
    • Adım atlanır kontrolü: threshold=None → o adım çalışmaz

    Method chaining kullanımı:
        selector = (
            FeatureSelector(target_col='score', task='regression')
            .remove_low_variance(threshold=0.01)
            .remove_high_correlation(threshold=0.95)
            .select_by_mutual_info(k=20)
        )
        selector.fit(X_train, y_train)
        X_train_clean = selector.transform(X_train)
        X_test_clean  = selector.transform(X_test)

    GridSearchCV kullanımı:
        pipe = Pipeline([
            ('selector', FeatureSelector(target_col='score')),
            ('model', Ridge())
        ])
        param_grid = {
            'selector__correlation_threshold': [0.85, 0.90, 0.95],
            'selector__mi_top_k': [10, 15, 20],
        }
        GridSearchCV(pipe, param_grid, cv=5).fit(X_train, y_train)
    """

    def __init__(
        self,
        target_col: str,
        task: str = "regression",
        variance_threshold: float = None,     # None → adım atlanır
        correlation_threshold: float = None,  # None → adım atlanır
        mi_top_k: int = None,                 # None → sıralar ama atmaz
    ):
        self.target_col = target_col
        self.task = task
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mi_top_k = mi_top_k

    # ------------------------------------------------------------------
    # Method Chaining API
    # __init__ parametrelerini günceller → clone() bozulmaz
    # ------------------------------------------------------------------

    def remove_low_variance(self, threshold: float = 0.01) -> "FeatureSelector":
        """Neredeyse sabit sütunları atar (variance ≤ threshold)."""
        self.variance_threshold = threshold
        return self

    def remove_high_correlation(self, threshold: float = 0.95) -> "FeatureSelector":
        """Birbiriyle yüksek korelasyonlu sütunlardan hedefe uzak olanı atar."""
        self.correlation_threshold = threshold
        return self

    def select_by_mutual_info(self, k: int = None) -> "FeatureSelector":
        """Mutual Information ile en bilgilendirici k özelliği seçer."""
        self.mi_top_k = k
        return self

    # ------------------------------------------------------------------
    # Yardımcı
    # ------------------------------------------------------------------

    def _numeric_cols(self, X: pd.DataFrame) -> list:
        return X.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    # Fit — sadece X_train görür
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureSelector":
        """
        Hangi sütunların atılacağını SADECE eğitim verisi üzerinden öğrenir.
        Test setine hiç dokunmaz → data leakage yok.

        Fitted attribute'lar (sklearn convention: trailing underscore):
            features_to_drop_  : atılacak sütun listesi
            mi_scores_         : mutual information skorları (inspect için)
        """
        # target_col X içindeyse ayır, dışarıdan y verilmişse onu kullan
        if y is None:
            if self.target_col not in X.columns:
                raise ValueError(
                    f"'{self.target_col}' sütunu X içinde yok ve y de verilmedi."
                )
            y = X[self.target_col]

        # Çalışma kopyası — sadece feature sütunları
        X_work = X.drop(columns=[self.target_col], errors="ignore").copy()
        self.feature_names_in_ = np.array(X_work.columns.tolist())  # fit sırasındaki sütunlar
        dropped: set = set()

        # ── Adım 1: Düşük Varyans ──────────────────────────────────────
        if self.variance_threshold is not None:
            num_cols = self._numeric_cols(X_work)
            if num_cols:
                vt = VarianceThreshold(threshold=self.variance_threshold)
                vt.fit(X_work[num_cols])
                low_var = [
                    col for col, keep in zip(num_cols, vt.get_support())
                    if not keep
                ]
                dropped.update(low_var)
                X_work.drop(columns=low_var, inplace=True)
                msg = f"atıldı: {low_var}" if low_var else "atılan sütun yok"
                print(f"✅ Düşük varyans (≤{self.variance_threshold}) → {msg}")

        # ── Adım 2: Yüksek Korelasyon ─────────────────────────────────
        if self.correlation_threshold is not None:
            num_cols = self._numeric_cols(X_work)
            if len(num_cols) >= 2:
                corr_matrix = X_work[num_cols].corr().abs()
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                # Hedefe korelasyonu düşük olanı at (rastgele değil, bilinçli)
                target_corr = X_work[num_cols].corrwith(y).abs()

                to_drop: set = set()
                for col in upper.columns:
                    if col in to_drop:
                        continue
                    high_partners = upper.index[
                        upper[col] > self.correlation_threshold
                    ].tolist()
                    for partner in high_partners:
                        if partner in to_drop:
                            continue
                        loser = (
                            col
                            if target_corr.get(col, 0) <= target_corr.get(partner, 0)
                            else partner
                        )
                        to_drop.add(loser)
                        if loser == col:
                            break  # col elendi, diğer partnerleriyle kıyaslamak anlamsız

                dropped.update(to_drop)
                X_work.drop(columns=list(to_drop), errors="ignore", inplace=True)
                msg = f"atıldı: {list(to_drop)}" if to_drop else "atılan sütun yok"
                print(f"✅ Yüksek korelasyon (>{self.correlation_threshold}) → {msg}")

        # ── Adım 3: Mutual Information ────────────────────────────────
        num_cols = self._numeric_cols(X_work)
        if num_cols:
            mi_func = (
                mutual_info_regression if self.task == "regression"
                else mutual_info_classif
            )
            mi_scores = mi_func(
                X_work[num_cols].fillna(0), y, random_state=42
            )
            self.mi_scores_ = pd.Series(
                mi_scores, index=num_cols
            ).sort_values(ascending=False)

            print(f"\n📊 MI Skoru (ilk {min(10, len(self.mi_scores_))}):")
            print(self.mi_scores_.head(10).to_string())

            if self.mi_top_k is not None:
                keep = self.mi_scores_.head(self.mi_top_k).index.tolist()
                mi_drop = [c for c in num_cols if c not in keep]
                dropped.update(mi_drop)
                print(f"\n✅ MI filtresi (top {self.mi_top_k}) → atıldı: {mi_drop}")
        else:
            self.mi_scores_ = pd.Series(dtype=float)

        # set olarak sakla → transform'da duplicate drop riski yok
        self.features_to_drop_: set = dropped
        return self

    # ------------------------------------------------------------------
    # Transform — X_train ve X_test'e uygulanır
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit'te öğrenilen eleme listesini veri setine uygular.
        X_train ve X_test için aynı sütunları atar.
        """
        if not hasattr(self, "features_to_drop_"):
            raise RuntimeError("Önce .fit() çağırın.")

        cols_to_drop = [c for c in self.features_to_drop_ if c in X.columns]
        return X.drop(columns=cols_to_drop)  # orijinal index korunur

    # ------------------------------------------------------------------
    # Rapor
    # ------------------------------------------------------------------

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Fit sonrası kalan sütun isimlerini döner.
        sklearn ColumnTransformer ve FeatureUnion ile uyum için gerekli.
        input_features=None → fit sırasında görülen sütunlar kullanılır.
        """
        if not hasattr(self, "features_to_drop_"):
            raise RuntimeError("Önce .fit() çağırın.")
        features = (
            self.feature_names_in_
            if input_features is None
            else np.array(input_features)
        )
        return np.array([f for f in features if f not in self.features_to_drop_])

    def report(self):
        if not hasattr(self, "features_to_drop_"):
            print("⚠️  Henüz fit edilmedi.")
            return
        print(f"\n🗑️  Toplam atılan sütun: {len(self.features_to_drop_)}")
        for f in sorted(self.features_to_drop_):
            print(f"   – {f}")
        if hasattr(self, "mi_scores_") and not self.mi_scores_.empty:
            print(f"\n📊 Kalan özellikler (MI sırasıyla):")
            remaining = [
                c for c in self.mi_scores_.index
                if c not in self.features_to_drop_
            ]
            print(self.mi_scores_[remaining].to_string())
