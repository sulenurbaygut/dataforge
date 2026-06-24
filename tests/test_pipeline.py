"""
Temel entegrasyon testleri.
Çalıştırmak için: pytest tests/
"""
import numpy as np
import pandas as pd
import pytest

from src.cleaner import DataCleaner
from src.selector import FeatureSelector
from src.transformer import DataTransformer


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    base = np.random.randn(n)
    return pd.DataFrame({
        "num_a":       np.random.randn(n),
        "num_b":       np.random.randn(n),
        "low_var":     np.ones(n) * 0.5 + np.random.randn(n) * 1e-5,  # neredeyse sabit
        "high_corr":   base + np.random.randn(n) * 0.01,               # base ile ~1 korelasyon
        "base":        base,
        "cat_col":     np.random.choice(["A", "B", "C"], n),
        "target":      np.random.randn(n),
    })


# ── DataCleaner ───────────────────────────────────────────────────────────────

class TestDataCleaner:
    def test_drop_duplicates(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)
        result = DataCleaner(df_dup).drop_duplicates().get()
        assert len(result) == len(sample_df)

    def test_handle_missing_knn(self, sample_df):
        df = sample_df.copy()
        df.loc[:10, "num_a"] = np.nan
        result = DataCleaner(df).handle_missing(
            numeric_strategy="knn", exclude_cols=["target"]
        ).get()
        assert result["num_a"].isnull().sum() == 0

    def test_outlier_cap_no_data_loss(self, sample_df):
        before = len(sample_df)
        result = DataCleaner(sample_df).handle_outliers(
            method="iqr", action="cap", exclude_cols=["target"]
        ).get()
        assert len(result) == before

    def test_outlier_flag_adds_column(self, sample_df):
        result = DataCleaner(sample_df).handle_outliers(
            method="iqr", action="flag", exclude_cols=["target"],
            columns=["num_a"]
        ).get()
        assert "num_a_outlier" in result.columns

    def test_target_untouched_by_outlier(self, sample_df):
        original_target = sample_df["target"].copy()
        result = DataCleaner(sample_df).handle_outliers(
            method="iqr", action="cap", exclude_cols=["target"]
        ).get()
        pd.testing.assert_series_equal(result["target"], original_target)


# ── DataTransformer ───────────────────────────────────────────────────────────

class TestDataTransformer:
    def test_encode_onehot(self, sample_df):
        result = DataTransformer(sample_df).encode_categoricals(
            method="onehot", exclude_cols=["target"]
        ).get()
        assert "cat_col" not in result.columns
        assert any(c.startswith("cat_col_") for c in result.columns)

    def test_encode_label(self, sample_df):
        result = DataTransformer(sample_df).encode_categoricals(
            method="label", exclude_cols=["target"]
        ).get()
        assert result["cat_col"].dtype in [np.int32, np.int64, int]

    def test_scale_robust_excludes_target(self, sample_df):
        original_target = sample_df["target"].copy()
        result = DataTransformer(sample_df).scale_numerics(
            method="robust", exclude_cols=["target"]
        ).get()
        pd.testing.assert_series_equal(result["target"], original_target)

    def test_interaction_feature(self, sample_df):
        result = DataTransformer(sample_df).create_interaction_features(
            [("num_a", "num_b")]
        ).get()
        assert "num_a_x_num_b" in result.columns


# ── FeatureSelector ───────────────────────────────────────────────────────────

class TestFeatureSelector:
    def test_low_variance_removes_constant(self, sample_df):
        X = sample_df.drop(columns=["target", "cat_col"])
        y = sample_df["target"]
        selector = FeatureSelector(target_col="target").remove_low_variance(0.001)
        selector.fit(X, y)
        assert "low_var" in selector.features_to_drop_

    def test_high_corr_keeps_one(self, sample_df):
        X = sample_df.drop(columns=["target", "cat_col"])
        y = sample_df["target"]
        selector = FeatureSelector(target_col="target").remove_high_correlation(0.95)
        selector.fit(X, y)
        survivors = [c for c in ["base", "high_corr"] if c not in selector.features_to_drop_]
        assert len(survivors) == 1

    def test_target_not_dropped(self, sample_df):
        selector = FeatureSelector(target_col="target").remove_low_variance(0.001)
        selector.fit(sample_df)
        assert "target" not in selector.features_to_drop_

    def test_transform_preserves_index(self, sample_df):
        X = sample_df.drop(columns=["target", "cat_col"])
        y = sample_df["target"]
        X.index = range(100, 100 + len(X))
        selector = FeatureSelector(target_col="target").remove_low_variance(0.001)
        selector.fit(X, y)
        result = selector.transform(X)
        assert list(result.index) == list(X.index)

    def test_clone_compatibility(self, sample_df):
        from sklearn.base import clone
        selector = (
            FeatureSelector(target_col="target")
            .remove_low_variance(0.01)
            .remove_high_correlation(0.95)
        )
        cloned = clone(selector)
        assert cloned.variance_threshold == 0.01
        assert cloned.correlation_threshold == 0.95

    def test_get_feature_names_out(self, sample_df):
        X = sample_df.drop(columns=["target", "cat_col"])
        y = sample_df["target"]
        selector = FeatureSelector(target_col="target").remove_low_variance(0.001)
        selector.fit(X, y)
        names = selector.get_feature_names_out()
        assert "low_var" not in names
        assert len(names) > 0

    def test_fit_transform_no_leakage(self, sample_df):
        """Test seti fit sırasında görülmemeli."""
        X = sample_df.drop(columns=["target", "cat_col"])
        y = sample_df["target"]
        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train = y.iloc[:150]

        selector = FeatureSelector(target_col="target").remove_low_variance(0.001)
        selector.fit(X_train, y_train)

        X_train_t = selector.transform(X_train)
        X_test_t  = selector.transform(X_test)

        assert X_train_t.shape[1] == X_test_t.shape[1]
