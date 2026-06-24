import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class DataTransformer:
    """
    Method chaining API ile çalışan dönüşüm motoru.

    Kullanım:
        df_t = (
            DataTransformer(df)
            .encode_categoricals(method='auto')
            .scale_numerics(method='robust')
            .get()
        )
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._encoders: dict = {}
        self._scalers: dict = {}

    # ------------------------------------------------------------------
    # 1. Kategorik Encoding
    # ------------------------------------------------------------------

    def encode_categoricals(
        self,
        method: str = "auto",
        target_col: str = None,
        max_cardinality_for_ohe: int = 15,
        exclude_cols: list = None,
    ) -> "DataTransformer":
        """
        Kategorik sütunları sayısala dönüştürür.

        method:
            'auto'   – Kardinaliteye göre otomatik seçim:
                       ≤ max_cardinality_for_ohe  →  One-Hot Encoding
                       >  max_cardinality_for_ohe →  Label Encoding
            'onehot' – Tümüne One-Hot Encoding
            'label'  – Tümüne Label Encoding
            'target' – Target Encoding (target_col zorunlu, data leakage riski var)
        """
        exclude_cols = exclude_cols or []
        cat_cols = [
            c
            for c in self.df.select_dtypes(include=["object", "category"]).columns
            if c not in exclude_cols
        ]

        if not cat_cols:
            print("ℹ️  Encode edilecek kategorik sütun bulunamadı.")
            return self

        for col in cat_cols:
            cardinality = self.df[col].nunique()
            chosen = method

            if method == "auto":
                chosen = "onehot" if cardinality <= max_cardinality_for_ohe else "label"

            if chosen == "target":
                if target_col is None or target_col not in self.df.columns:
                    raise ValueError("Target Encoding için geçerli bir 'target_col' gerekli.")
                means = self.df.groupby(col)[target_col].mean()
                self.df[col] = self.df[col].map(means)
                self._encoders[col] = ("target", means)

            elif chosen == "onehot":
                dummies = pd.get_dummies(
                    self.df[col], prefix=col, drop_first=True, dtype=int
                )
                self.df = pd.concat(
                    [self.df.drop(columns=[col]), dummies], axis=1
                )
                self._encoders[col] = ("onehot", dummies.columns.tolist())

            else:  # label
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self._encoders[col] = ("label", le)

        print(
            f"✅ Encoding tamamlandı | "
            f"{len(cat_cols)} sütun işlendi | Yöntem: {method}"
        )
        return self

    # ------------------------------------------------------------------
    # 2. Sayısal Scaling
    # ------------------------------------------------------------------

    def scale_numerics(
        self,
        method: str = "robust",
        exclude_cols: list = None,
    ) -> "DataTransformer":
        """
        Sayısal sütunları ölçeklendirir.

        method:
            'standard' – StandardScaler: mean=0, std=1 (normal dağılım varsayımı)
            'minmax'   – MinMaxScaler: 0-1 aralığı (aykırı değere duyarlı)
            'robust'   – RobustScaler: medyan tabanlı, aykırı değere dayanıklı ✅
        """
        exclude_cols = exclude_cols or []
        num_cols = [
            c
            for c in self.df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

        if not num_cols:
            print("ℹ️  Scale edilecek sayısal sütun bulunamadı.")
            return self

        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        if method not in scalers:
            raise ValueError(f"Geçersiz scaling yöntemi: '{method}'")

        scaler = scalers[method]
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        self._scalers["numeric"] = (method, scaler, num_cols)

        print(
            f"✅ Scaling tamamlandı | "
            f"{len(num_cols)} sütun | Yöntem: {method}"
        )
        return self

    # ------------------------------------------------------------------
    # 3. Özellik Mühendisliği
    # ------------------------------------------------------------------

    def create_interaction_features(
        self, col_pairs: list
    ) -> "DataTransformer":
        """
        Belirtilen sütun çiftleri için çarpım özelliği üretir.
        Örn: [('study_hours', 'attendance')] → 'study_hours_x_attendance'
        Bu özellikler regresyon modellerinde doğrusal olmayan ilişkileri yakalar.
        """
        for col1, col2 in col_pairs:
            if col1 not in self.df.columns or col2 not in self.df.columns:
                print(f"⚠️  '{col1}' veya '{col2}' sütunu bulunamadı, atlanıyor.")
                continue
            new_col = f"{col1}_x_{col2}"
            self.df[new_col] = self.df[col1] * self.df[col2]
            print(f"✅ Interaction feature: {new_col}")
        return self

    def create_ratio_features(
        self, col_pairs: list, epsilon: float = 1e-8
    ) -> "DataTransformer":
        """
        Belirtilen sütun çiftleri için oran (ratio) özelliği üretir.
        Örn: [('correct', 'total')] → 'correct_per_total'
        epsilon: sıfıra bölmeyi engeller.
        """
        for numerator, denominator in col_pairs:
            if numerator not in self.df.columns or denominator not in self.df.columns:
                print(f"⚠️  Sütun bulunamadı: {numerator} / {denominator}, atlanıyor.")
                continue
            new_col = f"{numerator}_per_{denominator}"
            self.df[new_col] = self.df[numerator] / (self.df[denominator] + epsilon)
            print(f"✅ Ratio feature: {new_col}")
        return self

    # ------------------------------------------------------------------
    # 4. Çıktı
    # ------------------------------------------------------------------

    def get(self) -> pd.DataFrame:
        return self.df.reset_index(drop=True)

    @property
    def encoders(self) -> dict:
        """Eğitilmiş encoder nesnelerine erişim (test seti için)."""
        return self._encoders

    @property
    def scalers(self) -> dict:
        """Eğitilmiş scaler nesnelerine erişim (test seti için)."""
        return self._scalers
