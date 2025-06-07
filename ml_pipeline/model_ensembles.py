# ── ml_pipeline/model_ensembles.py ────────────────────────────────────────────
# Módulo de ensembles avanzados para machine learning
import os
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Optional, Union, List
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.base import BaseEstimator
import optuna
import joblib
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost no está disponible.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("LightGBM no está disponible.")


class ModelEnsembleFactory:
    def __init__(self, task_type: str = 'regression', random_state: int = 42):
        self.task_type = task_type.lower()
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.feature_names: Optional[List[str]] = None
        if self.task_type not in ['regression', 'classification']:
            raise ValueError("task_type debe ser 'regression' o 'classification'")

    def set_feature_names(self, feature_names: List[str]):
        """Guarda columnas para reindexar al evaluar."""
        self.feature_names = feature_names

    # ---------- Parámetros por defecto ----------
    def _get_default_rf_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'n_jobs': -1
        }

    def _get_default_xgb_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'n_jobs': -1
        }

    def _get_default_lgb_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'num_leaves': 31,
            'n_jobs': -1,
            'verbose': -1
        }

    # ---------- Modelos individuales ----------
    def create_individual_models(self,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 optimize_hyperparams: bool = True,
                                 n_trials: int = 100
                                 ) -> Dict[str, Any]:
        X_arr, y_arr = X_train.values, y_train.values

        # RandomForest
        rf_params = (self._optimize_random_forest(X_arr, y_arr, n_trials)
                     if optimize_hyperparams else self._get_default_rf_params())
        cls_rf = RandomForestRegressor if self.task_type == 'regression' else RandomForestClassifier
        self.models['random_forest'] = cls_rf(**rf_params, random_state=self.random_state)

        # XGBoost
        if XGB_AVAILABLE:
            xgb_params = (self._optimize_xgboost(X_arr, y_arr, n_trials)
                          if optimize_hyperparams else self._get_default_xgb_params())
            cls_xgb = xgb.XGBRegressor if self.task_type == 'regression' else xgb.XGBClassifier
            self.models['xgboost'] = cls_xgb(**xgb_params, random_state=self.random_state)

        # LightGBM
        if LGB_AVAILABLE:
            lgb_params = (self._optimize_lightgbm(X_arr, y_arr, n_trials)
                          if optimize_hyperparams else self._get_default_lgb_params())
            cls_lgb = lgb.LGBMRegressor if self.task_type == 'regression' else lgb.LGBMClassifier
            self.models['lightgbm'] = cls_lgb(**lgb_params, random_state=self.random_state)

        for m in self.models.values():
            m.fit(X_arr, y_arr)

        return self.models

    # ---------- Ensemble optimizado ----------
    def create_optimized_ensemble(self,
                                  X_train: Union[pd.DataFrame, np.ndarray],
                                  y_train: Union[pd.Series, np.ndarray],
                                  voting: str = 'soft'
                                  ) -> Any:
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_arr = X_train.values
        else:
            X_arr = X_train
        y_arr = y_train.values if hasattr(y_train, 'values') else y_train

        if not self.models:
            df = pd.DataFrame(X_arr, columns=self.feature_names) if self.feature_names else pd.DataFrame(X_arr)
            self.create_individual_models(df, pd.Series(y_arr), optimize_hyperparams=False)

        weights = self.get_ensemble_weights(pd.DataFrame(X_arr), pd.Series(y_arr))
        estimators = list(self.models.items())
        w_list = [weights.get(n, 1.0) for n in self.models]

        if self.task_type == 'regression':
            ensemble = VotingRegressor(estimators=estimators, weights=w_list)
        else:
            ensemble = VotingClassifier(estimators=estimators, voting=voting, weights=w_list)

        ensemble.fit(X_arr, y_arr)
        return ensemble

    def get_ensemble_weights(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series
                             ) -> Dict[str, float]:
        return self._calculate_diversity_weights(X_train, y_train)

    def _calculate_diversity_weights(self,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series
                                     ) -> Dict[str, float]:
        n_models = len(self.models)
        base = 1.0 / n_models
        weights = {}
        for name in self.models:
            if name in ('xgboost', 'lightgbm'):
                weights[name] = base * 1.1
            elif name == 'random_forest':
                weights[name] = base * 0.9
            else:
                weights[name] = base
        total = sum(weights.values())
        return {n: w/total for n, w in weights.items()}

    # ---------- Evaluación ----------
    def evaluate_ensemble_performance(self,
                                      ensemble: Any,
                                      X_test: Union[pd.DataFrame, np.ndarray],
                                      y_test: pd.Series,
                                      individual_models: bool = True
                                      ) -> Dict[str, Dict[str, float]]:
        # Prepara X_arr
        if isinstance(X_test, pd.DataFrame) and self.feature_names:
            X_arr = X_test.reindex(columns=self.feature_names, fill_value=0).values
        else:
            X_arr = X_test.values if hasattr(X_test, 'values') else X_test
        y_arr = y_test.values if hasattr(y_test, 'values') else y_test

        # Función interna para ajustar dimensiones por modelo
        def align(X: np.ndarray, model) -> np.ndarray:
            n_req = getattr(model, 'n_features_in_', X.shape[1])
            if X.shape[1] < n_req:
                # pad zeros
                pad = np.zeros((X.shape[0], n_req - X.shape[1]))
                return np.hstack([X, pad])
            elif X.shape[1] > n_req:
                return X[:, :n_req]
            return X

        results: Dict[str, Dict[str, float]] = {}

        # Ensemble
        X_e = align(X_arr, ensemble)
        y_pred = ensemble.predict(X_e)
        if self.task_type == 'regression':
            results['ensemble'] = {
                'mse': mean_squared_error(y_arr, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_arr, y_pred)),
                'r2': r2_score(y_arr, y_pred)
            }
        else:
            results['ensemble'] = {'accuracy': accuracy_score(y_arr, y_pred)}

        # Modelos individuales
        if individual_models:
            for name, model in self.models.items():
                X_m = align(X_arr, model)
                y_pred_i = model.predict(X_m)
                if self.task_type == 'regression':
                    results[name] = {
                        'mse': mean_squared_error(y_arr, y_pred_i),
                        'rmse': np.sqrt(mean_squared_error(y_arr, y_pred_i)),
                        'r2': r2_score(y_arr, y_pred_i)
                    }
                else:
                    results[name] = {'accuracy': accuracy_score(y_arr, y_pred_i)}

        return results

    # ---------- Guardado y carga ----------
    def save_evaluation_results(self,
                                results: Dict[str, Dict[str, float]],
                                filepath: str):
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(filepath)
        ax = df.plot(
            kind='bar', subplots=True,
            layout=(1, len(df.columns)),
            figsize=(6 * len(df.columns), 4)
        )
        plt.tight_layout()
        fig_path = os.path.splitext(filepath)[0] + '_plots.png'
        plt.savefig(fig_path)
        plt.close()
        print(f"✅ Resultados guardados en {filepath} y gráficos en {fig_path}")

    def load_ensemble(self, filepath: str) -> Dict[str, Any]:
        data = joblib.load(filepath)
        self.task_type = data.get('task_type', self.task_type)
        self.ensemble_weights = data.get('weights', {})
        self.models = data.get('individual_models', {})
        print(f"✅ Ensemble cargado desde {filepath}")
        return data

    # ---------- Optimización privada ----------
    def _optimize_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int) -> Dict[str, Any]:
        # ... implementación existente ...
        pass

    def _optimize_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int) -> Dict[str, Any]:
        # ... implementación existente ...
        pass

    def _optimize_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int) -> Dict[str, Any]:
        # ... implementación existente ...
        pass


class CustomEnsemble(BaseEstimator):
    """
    Ensemble personalizado con capacidades avanzadas
    """
    def __init__(self,
                 models: Dict[str, Any],
                 weights: Optional[Dict[str, float]] = None,
                 method: str = 'weighted_avg'):
        self.models = models
        self.weights = weights or {n: 1.0 for n in models}
        self.method = method
        self.is_fitted = False

    def fit(self, X, y):
        for m in self.models.values():
            m.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("El ensemble debe ser entrenado primero")
        preds = np.array([m.predict(X) for m in self.models.values()])
        w = np.array([self.weights[n] for n in self.models])
        if self.method == 'weighted_avg':
            return np.average(preds, axis=0, weights=w)
        elif self.method == 'median':
            return np.median(preds, axis=0)
        elif self.method == 'max':
            return np.max(preds, axis=0)
        else:
            return np.mean(preds, axis=0)