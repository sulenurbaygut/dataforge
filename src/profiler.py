import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DataProfiler:
    """
    Veri setinin tam profilini çıkarır:
    eksik değer yüzdeleri, dağılımlar, skewness, korelasyonlar.
    Tüm grafikler outputs/ klasörüne kaydedilir.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "outputs"):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def summary(self) -> pd.DataFrame:
        """Her sütun için eksik değer, unique, skewness ve istatistik raporu."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns

        report = pd.DataFrame(
            {
                "dtype": self.df.dtypes,
                "missing_count": self.df.isnull().sum(),
                "missing_pct": (
                    self.df.isnull().sum() / len(self.df) * 100
                ).round(2),
                "unique_count": self.df.nunique(),
                "unique_pct": (
                    self.df.nunique() / len(self.df) * 100
                ).round(2),
            }
        )

        for col in numeric_cols:
            report.loc[col, "mean"] = round(self.df[col].mean(), 4)
            report.loc[col, "std"] = round(self.df[col].std(), 4)
            report.loc[col, "skewness"] = round(self.df[col].skew(), 4)

        print("=" * 65)
        print("📊  VERİ PROFİLİ")
        print("=" * 65)
        print(f"  Toplam satır      : {self.df.shape[0]:,}")
        print(f"  Toplam sütun      : {self.df.shape[1]}")
        print(f"  Sayısal sütunlar  : {len(numeric_cols)}")
        print(f"  Kategorik sütunlar: {len(categorical_cols)}")
        print(
            f"  Toplam eksik değer: {self.df.isnull().sum().sum():,}"
            f" ({self.df.isnull().sum().sum() / self.df.size * 100:.2f}%)"
        )
        print("=" * 65)
        print(report.to_string())
        return report

    def target_analysis(self, target_col: str) -> pd.Series:
        """Hedef değişkenle korelasyonu yüksek olan özellikleri sıralar ve görselleştirir."""
        if target_col not in self.df.columns:
            raise ValueError(f"'{target_col}' sütunu bulunamadı.")

        numeric_df = self.df.select_dtypes(include=[np.number])
        correlations = (
            numeric_df.corr()[target_col]
            .drop(target_col, errors="ignore")
            .sort_values(key=abs, ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, max(5, len(correlations) * 0.4)))
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in correlations.values]
        ax.barh(correlations.index, correlations.values, color=colors, edgecolor="white")
        ax.set_title(
            f"'{target_col}' ile Korelasyon (Önem Sırası)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Korelasyon Katsayısı")
        ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
        plt.tight_layout()

        path = self.output_dir / "target_correlation.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊  Hedef korelasyon grafiği kaydedildi: {path}")
        return correlations

    def correlation_heatmap(self):
        """Tüm sayısal sütunlar arası korelasyon matrisini görselleştirir."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            print("ℹ️  Korelasyon için yeterli sayısal sütun yok.")
            return

        mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
        fig, ax = plt.subplots(figsize=(max(8, numeric_df.shape[1]), max(6, numeric_df.shape[1] - 1)))
        sns.heatmap(
            numeric_df.corr(),
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax,
            linewidths=0.5,
            vmin=-1,
            vmax=1,
        )
        ax.set_title("Korelasyon Matrisi", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊  Korelasyon haritası kaydedildi: {path}")

    def distribution_plots(self, cols: list = None, max_cols: int = 12):
        """Sayısal sütunların histogramlarını ve KDE eğrilerini üretir."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        cols = cols or numeric_df.columns[:max_cols].tolist()
        n = len(cols)

        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
        axes = axes.flatten() if n > 1 else [axes]

        for i, col in enumerate(cols):
            data = self.df[col].dropna()
            axes[i].hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
            ax2 = axes[i].twinx()
            data.plot(kind="kde", ax=ax2, color="orange", linewidth=2)
            ax2.set_ylabel("")
            ax2.set_yticks([])
            axes[i].set_title(f"{col}\nskew={data.skew():.2f}", fontsize=9)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Özellik Dağılımları", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        path = self.output_dir / "distributions.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊  Dağılım grafikleri kaydedildi: {path}")
