from pathlib import Path

import pandas as pd

from .cleaner import DataCleaner
from .loader import DataLoader
from .profiler import DataProfiler
from .selector import FeatureSelector
from .transformer import DataTransformer


class DataPipeline:
    """
    Uçtan uca veri ön işleme pipeline'ı.

    Load → Profile → Clean → Transform → Select → Save

    Config referansı:
    {
        'file_path':             'data/raw/train.csv',
        'target_col':            'score',
        'task':                  'regression',   # 'regression' | 'classification'

        # Temizlik
        'missing_strategy':      'knn',          # 'mean' | 'median' | 'knn' | 'iterative'
        'outlier_method':        'iqr',          # 'iqr' | 'zscore' | 'isolation_forest'
        'outlier_action':        'cap',          # 'cap' | 'remove' | 'flag'

        # Dönüşüm
        'encoding_method':       'auto',         # 'auto' | 'onehot' | 'label' | 'target'
        'scaling_method':        'robust',       # 'standard' | 'minmax' | 'robust'
        'exclude_from_scaling':  [],
        'exclude_from_encoding': [],
        'interaction_pairs':     [],             # [('col1', 'col2'), ...]
        'ratio_pairs':           [],             # [('numerator', 'denominator'), ...]

        # Özellik seçimi
        'run_feature_selection': True,
        'variance_threshold':    0.01,           # None → adım atlanır
        'corr_threshold':        0.95,           # None → adım atlanır
        'mi_top_k':              None,           # None → sıralar ama atmaz

        # Çıktı
        'output_path':           'data/processed/clean_data.csv',
    }
    """

    def __init__(self, config: dict):
        self.config = config
        self.df_raw: pd.DataFrame = None
        self.df_processed: pd.DataFrame = None
        self.selector: FeatureSelector = None

    def run(self, profile: bool = True) -> pd.DataFrame:
        c = self.config
        target = c.get("target_col")
        exclude = [target] if target else []

        print("\n" + "=" * 65)
        print("🚀  DATA PIPELINE BAŞLATILDI")
        print("=" * 65)

        # ── 1. LOAD ──────────────────────────────────────────────────
        self.df_raw = DataLoader(c["file_path"]).load()

        # ── 2. PROFILE ───────────────────────────────────────────────
        if profile:
            profiler = DataProfiler(self.df_raw, output_dir="outputs")
            profiler.summary()
            if target and target in self.df_raw.columns:
                profiler.target_analysis(target)
            profiler.correlation_heatmap()
            profiler.distribution_plots()

        # ── 3. CLEAN ─────────────────────────────────────────────────
        df_clean = (
            DataCleaner(self.df_raw)
            .drop_duplicates()
            .handle_missing(
                numeric_strategy=c.get("missing_strategy", "knn"),
                exclude_cols=exclude,
            )
            .handle_outliers(
                method=c.get("outlier_method", "iqr"),
                action=c.get("outlier_action", "cap"),
                exclude_cols=exclude,
            )
            .get()
        )

        # ── 4. TRANSFORM ─────────────────────────────────────────────
        exclude_enc = list(set(c.get("exclude_from_encoding", []) + exclude))
        exclude_sc  = list(set(c.get("exclude_from_scaling",  []) + exclude))

        transformer = DataTransformer(df_clean)
        transformer.encode_categoricals(
            method=c.get("encoding_method", "auto"),
            target_col=target,
            exclude_cols=exclude_enc,
        )
        if c.get("interaction_pairs"):
            transformer.create_interaction_features(c["interaction_pairs"])
        if c.get("ratio_pairs"):
            transformer.create_ratio_features(c["ratio_pairs"])
        transformer.scale_numerics(
            method=c.get("scaling_method", "robust"),
            exclude_cols=exclude_sc,
        )
        df_transformed = transformer.get()

        # ── 5. FEATURE SELECTION ─────────────────────────────────────
        if c.get("run_feature_selection", False) and target:
            self.selector = (
                FeatureSelector(
                    target_col=target,
                    task=c.get("task", "regression"),
                    variance_threshold=c.get("variance_threshold", 0.01),
                    correlation_threshold=c.get("corr_threshold", 0.95),
                    mi_top_k=c.get("mi_top_k"),
                )
            )
            # fit: sadece eğitim verisini görür (target izole edilir)
            self.selector.fit(df_transformed)
            self.selector.report()
            df_transformed = self.selector.transform(df_transformed)

        # ── 6. SAVE ──────────────────────────────────────────────────
        output_path = Path(c.get("output_path", "data/processed/clean_data.csv"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_transformed.to_csv(output_path, index=False)

        self.df_processed = df_transformed

        print("\n" + "=" * 65)
        print("✅  PIPELINE TAMAMLANDI")
        print(f"   Ham → Temiz : {self.df_raw.shape}  →  {df_transformed.shape}")
        print(f"   Çıktı       : {output_path}")
        print("=" * 65 + "\n")

        return df_transformed
