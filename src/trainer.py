import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class ModelTrainer:
    """
    Model karşılaştırma, Optuna hyperparameter tuning ve ensemble motoru.

    Kullanım:
        trainer = ModelTrainer(task='regression', cv=5)
        results  = trainer.compare_models(X_train, y_train)
        best     = trainer.tune(X_train, y_train, model_name='lgbm', n_trials=100)
        ensemble = trainer.build_ensemble(X_train, y_train, top_n=3)
        metrics  = trainer.evaluate(ensemble, X_test, y_test)
    """

    def __init__(
        self,
        task: str = "regression",
        cv: int = 5,
        random_state: int = 42,
        output_dir: str = "outputs",
    ):
        self.task = task
        self.cv = cv
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.comparison_results_: pd.DataFrame = None
        self.best_model_name_: str = None
        self.tuned_model_ = None
        self.ensemble_ = None

    # ------------------------------------------------------------------
    # 1. Model Kataloğu
    # ------------------------------------------------------------------

    def _get_models(self) -> dict:
        """Görev tipine göre tüm modelleri döner."""
        if self.task == "regression":
            models = {
                "ridge":    Ridge(random_state=self.random_state)          if hasattr(Ridge, 'random_state') else Ridge(),
                "lasso":    Lasso(random_state=self.random_state)          if hasattr(Lasso, 'random_state') else Lasso(),
                "elasticnet": ElasticNet(random_state=self.random_state)   if hasattr(ElasticNet, 'random_state') else ElasticNet(),
                "svr":      Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
                "rf":       RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                "gbm":      GradientBoostingRegressor(random_state=self.random_state),
            }
        else:
            models = {
                "logistic": LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=-1),
                "svc":      Pipeline([("scaler", StandardScaler()), ("model", SVC(probability=True))]),
                "rf":       RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                "gbm":      GradientBoostingClassifier(random_state=self.random_state),
            }

        if XGB_AVAILABLE:
            if self.task == "regression":
                models["xgb"] = xgb.XGBRegressor(
                    n_estimators=100, random_state=self.random_state,
                    verbosity=0, n_jobs=-1
                )
            else:
                models["xgb"] = xgb.XGBClassifier(
                    n_estimators=100, random_state=self.random_state,
                    verbosity=0, n_jobs=-1, eval_metric="logloss"
                )

        if LGB_AVAILABLE:
            if self.task == "regression":
                models["lgbm"] = lgb.LGBMRegressor(
                    n_estimators=100, random_state=self.random_state,
                    verbose=-1, n_jobs=-1
                )
            else:
                models["lgbm"] = lgb.LGBMClassifier(
                    n_estimators=100, random_state=self.random_state,
                    verbose=-1, n_jobs=-1
                )

        return models

    def _get_cv_strategy(self, y):
        if self.task == "regression":
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

    def _scoring(self) -> str:
        return "neg_root_mean_squared_error" if self.task == "regression" else "f1_weighted"

    # ------------------------------------------------------------------
    # 2. Model Karşılaştırma
    # ------------------------------------------------------------------

    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Tüm modelleri cross-validation ile karşılaştırır.
        Sonuçları DataFrame ve grafik olarak kaydeder.
        """
        models = self._get_models()
        cv_strategy = self._get_cv_strategy(y)
        scoring = self._scoring()

        print("\n" + "=" * 60)
        print("📊  MODEL KARŞILAŞTIRMA BAŞLADI")
        print("=" * 60)

        rows = []
        for name, model in models.items():
            start = time.time()
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

            if "neg_" in scoring:
                scores = -scores  # pozitife çevir

            elapsed = time.time() - start
            row = {
                "model":    name,
                "mean":     round(scores.mean(), 4),
                "std":      round(scores.std(), 4),
                "min":      round(scores.min(), 4),
                "max":      round(scores.max(), 4),
                "time_s":   round(elapsed, 2),
            }
            rows.append(row)
            metric = "RMSE" if self.task == "regression" else "F1"
            print(f"  {name:<12} {metric}: {row['mean']:.4f} ± {row['std']:.4f}  ({elapsed:.1f}s)")

        results = pd.DataFrame(rows)

        # Regresyonda en düşük RMSE en iyi, sınıflandırmada en yüksek F1
        ascending = self.task == "regression"
        results = results.sort_values("mean", ascending=ascending).reset_index(drop=True)
        self.comparison_results_ = results
        self.best_model_name_ = results.iloc[0]["model"]

        print(f"\n🏆  En iyi model: {self.best_model_name_} ({results.iloc[0]['mean']:.4f})")

        # Grafik
        self._plot_comparison(results)

        # CSV kaydet
        path = self.output_dir / "model_comparison.csv"
        results.to_csv(path, index=False)
        print(f"💾  Sonuçlar kaydedildi: {path}")

        return results

    def _plot_comparison(self, results: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(results))]
        bars = ax.barh(results["model"], results["mean"], xerr=results["std"],
                       color=colors, edgecolor="white", capsize=4)
        metric = "RMSE (düşük = iyi)" if self.task == "regression" else "F1 Score (yüksek = iyi)"
        ax.set_title(f"Model Karşılaştırma — {metric}", fontweight="bold")
        ax.set_xlabel(metric)
        for bar, val in zip(bars, results["mean"]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9)
        plt.tight_layout()
        path = self.output_dir / "model_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊  Grafik kaydedildi: {path}")

    # ------------------------------------------------------------------
    # 3. Optuna Hyperparameter Tuning
    # ------------------------------------------------------------------

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = None,
        n_trials: int = 50,
    ):
        """
        Optuna ile seçilen modelin hiperparametrelerini optimize eder.
        model_name=None → karşılaştırmadaki en iyi model seçilir.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna kurulu değil: pip install optuna")

        model_name = model_name or self.best_model_name_
        if model_name is None:
            raise RuntimeError("Önce compare_models() çalıştırın.")

        print(f"\n🔍  Optuna Tuning — {model_name} ({n_trials} deneme)")

        cv_strategy = self._get_cv_strategy(y)
        scoring = self._scoring()

        def objective(trial):
            model = self._suggest_params(trial, model_name)
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
            return -scores.mean() if "neg_" not in scoring else scores.mean()

        direction = "minimize" if self.task == "regression" else "maximize"
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        print(f"✅  En iyi parametreler: {best_params}")
        print(f"✅  En iyi skor: {abs(study.best_value):.4f}")

        # En iyi parametrelerle modeli yeniden kur
        tuned = self._build_model_with_params(model_name, best_params)
        tuned.fit(X, y)
        self.tuned_model_ = tuned

        # Optuna görselleştirme
        self._plot_optuna(study, model_name)

        return tuned

    def _suggest_params(self, trial, model_name: str):
        """Her model için Optuna arama uzayı."""
        rs = self.random_state

        if model_name == "ridge":
            alpha = trial.suggest_float("alpha", 0.01, 100, log=True)
            return Ridge(alpha=alpha)

        elif model_name == "lasso":
            alpha = trial.suggest_float("alpha", 0.0001, 10, log=True)
            return Lasso(alpha=alpha, max_iter=5000)

        elif model_name == "elasticnet":
            alpha  = trial.suggest_float("alpha", 0.0001, 10, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

        elif model_name in ("svr", "svc"):
            C       = trial.suggest_float("C", 0.01, 100, log=True)
            kernel  = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
            gamma   = trial.suggest_categorical("gamma", ["scale", "auto"])
            if self.task == "regression":
                eps = trial.suggest_float("epsilon", 0.01, 1.0)
                return Pipeline([("scaler", StandardScaler()), ("model", SVR(C=C, kernel=kernel, gamma=gamma, epsilon=eps))])
            return Pipeline([("scaler", StandardScaler()), ("model", SVC(C=C, kernel=kernel, gamma=gamma, probability=True))])

        elif model_name == "rf":
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 500),
                "max_depth":     trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":  trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                "random_state":  rs, "n_jobs": -1,
            }
            return RandomForestRegressor(**params) if self.task == "regression" else RandomForestClassifier(**params)

        elif model_name == "gbm":
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 50, 500),
                "max_depth":       trial.suggest_int("max_depth", 2, 8),
                "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "random_state":    rs,
            }
            return GradientBoostingRegressor(**params) if self.task == "regression" else GradientBoostingClassifier(**params)

        elif model_name == "xgb" and XGB_AVAILABLE:
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 500),
                "max_depth":     trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":     trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda":    trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                "random_state":  rs, "verbosity": 0, "n_jobs": -1,
            }
            if self.task == "regression":
                return xgb.XGBRegressor(**params)
            return xgb.XGBClassifier(**params, eval_metric="logloss")

        elif model_name == "lgbm" and LGB_AVAILABLE:
            params = {
                "n_estimators":   trial.suggest_int("n_estimators", 50, 500),
                "max_depth":      trial.suggest_int("max_depth", 2, 10),
                "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves":     trial.suggest_int("num_leaves", 20, 200),
                "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":      trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda":     trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                "random_state":   rs, "verbose": -1, "n_jobs": -1,
            }
            return lgb.LGBMRegressor(**params) if self.task == "regression" else lgb.LGBMClassifier(**params)

        raise ValueError(f"Bilinmeyen model: {model_name}")

    def _build_model_with_params(self, model_name: str, params: dict):
        """Optuna'nın bulduğu parametrelerle sıfırdan model kurar."""
        params = {**params, "random_state": self.random_state} if "random_state" not in params else params
        if model_name == "ridge":
            return Ridge(**{k: v for k, v in params.items() if k in ["alpha"]})
        elif model_name == "lasso":
            return Lasso(**{k: v for k, v in params.items() if k in ["alpha"]}, max_iter=5000)
        elif model_name == "elasticnet":
            return ElasticNet(**{k: v for k, v in params.items() if k in ["alpha", "l1_ratio"]}, max_iter=5000)
        elif model_name in ("svr", "svc"):
            p = {k: v for k, v in params.items() if k in ["C", "kernel", "gamma", "epsilon"]}
            base = SVR(**p) if self.task == "regression" else SVC(**p, probability=True)
            return Pipeline([("scaler", StandardScaler()), ("model", base)])
        elif model_name == "rf":
            p = {k: v for k, v in params.items() if k in ["n_estimators","max_depth","min_samples_split","min_samples_leaf","max_features"]}
            cls = RandomForestRegressor if self.task == "regression" else RandomForestClassifier
            return cls(**p, random_state=self.random_state, n_jobs=-1)
        elif model_name == "gbm":
            p = {k: v for k, v in params.items() if k in ["n_estimators","max_depth","learning_rate","subsample","min_samples_leaf"]}
            cls = GradientBoostingRegressor if self.task == "regression" else GradientBoostingClassifier
            return cls(**p, random_state=self.random_state)
        elif model_name == "xgb" and XGB_AVAILABLE:
            cls = xgb.XGBRegressor if self.task == "regression" else xgb.XGBClassifier
            return cls(**params, verbosity=0, n_jobs=-1)
        elif model_name == "lgbm" and LGB_AVAILABLE:
            cls = lgb.LGBMRegressor if self.task == "regression" else lgb.LGBMClassifier
            return cls(**params, verbose=-1, n_jobs=-1)
        raise ValueError(f"Bilinmeyen model: {model_name}")

    def _plot_optuna(self, study, model_name: str):
        try:
            trials_df = study.trials_dataframe()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(trials_df["number"], trials_df["value"], alpha=0.5, color="steelblue")
            ax.plot(trials_df["number"],
                    trials_df["value"].cummin() if self.task == "regression" else trials_df["value"].cummax(),
                    color="red", linewidth=2, label="En iyi")
            ax.set_title(f"Optuna Optimizasyon Geçmişi — {model_name}", fontweight="bold")
            ax.set_xlabel("Deneme")
            ax.set_ylabel("Skor")
            ax.legend()
            plt.tight_layout()
            path = self.output_dir / f"optuna_{model_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"📊  Optuna grafiği: {path}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 4. Ensemble
    # ------------------------------------------------------------------

    def build_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 3,
        method: str = "stacking",  # 'stacking' | 'voting'
    ):
        """
        En iyi top_n modelden ensemble oluşturur.

        method:
            'stacking' – Meta-learner (Ridge) ile stacking (daha güçlü)
            'voting'   – Ağırlıklı ortalama (daha hızlı)
        """
        if self.comparison_results_ is None:
            raise RuntimeError("Önce compare_models() çalıştırın.")

        top_models_names = self.comparison_results_.head(top_n)["model"].tolist()
        all_models = self._get_models()
        estimators = [(name, all_models[name]) for name in top_models_names if name in all_models]

        print(f"\n🔗  Ensemble oluşturuluyor ({method}) — Modeller: {top_models_names}")

        if self.task == "regression":
            if method == "stacking":
                ensemble = StackingRegressor(
                    estimators=estimators,
                    final_estimator=Ridge(),
                    cv=self.cv,
                    n_jobs=-1,
                )
            else:
                ensemble = VotingRegressor(estimators=estimators, n_jobs=-1)
        else:
            if method == "stacking":
                ensemble = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=1000),
                    cv=self.cv,
                    n_jobs=-1,
                )
            else:
                ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

        ensemble.fit(X, y)
        self.ensemble_ = ensemble

        # CV skoru
        cv_strategy = self._get_cv_strategy(y)
        scores = cross_val_score(ensemble, X, y, cv=cv_strategy, scoring=self._scoring(), n_jobs=-1)
        score = abs(scores.mean())
        metric = "RMSE" if self.task == "regression" else "F1"
        print(f"✅  Ensemble CV {metric}: {score:.4f} ± {abs(scores.std()):.4f}")

        return ensemble

    # ------------------------------------------------------------------
    # 5. Değerlendirme
    # ------------------------------------------------------------------

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Test seti üzerinde model performansını değerlendirir."""
        y_pred = model.predict(X_test)

        print("\n" + "=" * 60)
        print("📋  TEST SETİ SONUÇLARI")
        print("=" * 60)

        if self.task == "regression":
            metrics = {
                "RMSE":  round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                "MAE":   round(mean_absolute_error(y_test, y_pred), 4),
                "R²":    round(r2_score(y_test, y_pred), 4),
            }
        else:
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = {
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "F1":       round(f1_score(y_test, y_pred, average="weighted"), 4),
                "AUC":      round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
            }

        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

        self._plot_predictions(y_test, y_pred)
        return metrics

    def _plot_predictions(self, y_test, y_pred):
        if self.task != "regression":
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Gerçek vs Tahmin
        axes[0].scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=20)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5)
        axes[0].set_xlabel("Gerçek")
        axes[0].set_ylabel("Tahmin")
        axes[0].set_title("Gerçek vs Tahmin", fontweight="bold")

        # Residuals
        residuals = y_test - y_pred
        axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
        axes[1].set_title("Hata Dağılımı (Residuals)", fontweight="bold")
        axes[1].set_xlabel("Hata")

        plt.tight_layout()
        path = self.output_dir / "prediction_analysis.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊  Tahmin grafiği: {path}")
