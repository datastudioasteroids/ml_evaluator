"""
ml_pipeline/model_ensembles.py
Módulo de ensembles avanzados para machine learning
Implementa estrategias de combinación, optimización de hiperparámetros y evaluación con gráficos.
"""
import os
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Optional, Union
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.base import BaseEstimator
import optuna
import joblib

# Imports condicionales de XGBoost y LightGBM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost no está disponible. Instala con: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("LightGBM no está disponible. Instala con: pip install lightgbm")

import matplotlib.pyplot as plt


class ModelEnsembleFactory:
    def __init__(self, task_type: str = 'regression', random_state: int = 42):
        self.task_type = task_type.lower()
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.ensemble_weights: Dict[str, float] = {}
        if self.task_type not in ['regression', 'classification']:
            raise ValueError("task_type debe ser 'regression' o 'classification'")

    # --------------------------------------------
    # MÉTODOS PARA PARÁMETROS POR DEFECTO
    # --------------------------------------------
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

    def create_individual_models(self,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 optimize_hyperparams: bool = True,
                                 n_trials: int = 100
                                 ) -> Dict[str, Any]:
        """Crea y entrena modelos base"""
        X_arr, y_arr = X_train.values, y_train.values

        # RandomForest
        rf_params = (self._optimize_random_forest(X_arr, y_arr, n_trials)
                     if optimize_hyperparams else
                     self._get_default_rf_params())
        if self.task_type == 'regression':
            self.models['random_forest'] = RandomForestRegressor(**rf_params,
                                                                 random_state=self.random_state)
        else:
            self.models['random_forest'] = RandomForestClassifier(**rf_params,
                                                                  random_state=self.random_state)

        # XGBoost
        if XGB_AVAILABLE:
            xgb_params = (self._optimize_xgboost(X_arr, y_arr, n_trials)
                          if optimize_hyperparams else
                          self._get_default_xgb_params())
            cls_xgb = xgb.XGBRegressor if self.task_type == 'regression' else xgb.XGBClassifier
            self.models['xgboost'] = cls_xgb(**xgb_params, random_state=self.random_state)

        # LightGBM
        if LGB_AVAILABLE:
            lgb_params = (self._optimize_lightgbm(X_arr, y_arr, n_trials)
                          if optimize_hyperparams else
                          self._get_default_lgb_params())
            cls_lgb = lgb.LGBMRegressor if self.task_type == 'regression' else lgb.LGBMClassifier
            self.models['lightgbm'] = cls_lgb(**lgb_params, random_state=self.random_state)

        # Entrenar todos los modelos
        for model in self.models.values():
            model.fit(X_arr, y_arr)

        return self.models

    def create_optimized_ensemble(self,
                                  X_train: Union[pd.DataFrame, np.ndarray],
                                  y_train: Union[pd.Series, np.ndarray],
                                  voting: str = 'soft'
                                  ) -> Any:
        """Crea y entrena un Voting ensemble optimizado"""
        # Conversión a arrays
        X_arr = X_train.values if hasattr(X_train, 'values') else X_train
        y_arr = y_train.values if hasattr(y_train, 'values') else y_train

        # Si no hay modelos pre-creados, los generamos sin optimizar
        if not self.models:
            self.create_individual_models(pd.DataFrame(X_arr),
                                          pd.Series(y_arr),
                                          optimize_hyperparams=False)

        # Obtener pesos según heurística de diversidad
        weights = self.get_ensemble_weights(pd.DataFrame(X_arr), pd.Series(y_arr))

        estimators = list(self.models.items())
        w_list = [weights.get(name, 1.0) for name in self.models]

        if self.task_type == 'regression':
            ensemble = VotingRegressor(estimators=estimators, weights=w_list)
        else:
            ensemble = VotingClassifier(estimators=estimators,
                                        voting=voting,
                                        weights=w_list)

        ensemble.fit(X_arr, y_arr)
        return ensemble

    def get_ensemble_weights(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series
                             ) -> Dict[str, float]:
        """
        Método puente: delega en la heurística de diversidad.
        """
        return self._calculate_diversity_weights(X_train, y_train)

    def _calculate_diversity_weights(self,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series
                                     ) -> Dict[str, float]:
        """Calcula pesos basados en diversidad de modelos"""
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
        return {n: w / total for n, w in weights.items()}

    # Métricas y guardado
    def evaluate_ensemble_performance(self,
                                      ensemble: Any,
                                      X_test: pd.DataFrame,
                                      y_test: pd.Series,
                                      individual_models: bool = True
                                      ) -> Dict[str, Dict[str, float]]:
        """Evalúa y retorna métricas"""
        X_arr, y_arr = X_test.values, y_test.values
        results: Dict[str, Dict[str, float]] = {}

        # Ensemble
        y_pred = ensemble.predict(X_arr)
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
                y_pred_i = model.predict(X_arr)
                if self.task_type == 'regression':
                    results[name] = {
                        'mse': mean_squared_error(y_arr, y_pred_i),
                        'rmse': np.sqrt(mean_squared_error(y_arr, y_pred_i)),
                        'r2': r2_score(y_arr, y_pred_i)
                    }
                else:
                    results[name] = {'accuracy': accuracy_score(y_arr, y_pred_i)}

        return results

    def save_evaluation_results(self,
                                results: Dict[str, Dict[str, float]],
                                filepath: str):
        """Guarda CSV y gráficos de métricas"""
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(filepath)
        ax = df.plot(
            kind='bar',
            subplots=True,
            layout=(1, len(df.columns)),
            figsize=(6 * len(df.columns), 4)
        )
        plt.tight_layout()
        fig_path = os.path.splitext(filepath)[0] + '_plots.png'
        plt.savefig(fig_path)
        plt.close()
        print(f"✅ Resultados guardados en {filepath} y gráficos en {fig_path}")

    def load_ensemble(self, filepath: str) -> Dict[str, Any]:
        """Carga ensemble y modelos desde archivo"""
        data = joblib.load(filepath)
        self.task_type = data['task_type']
        self.ensemble_weights = data.get('weights', {})
        self.models = data.get('individual_models', {})
        print(f"✅ Ensemble cargado desde {filepath}")
        return data

    # --------------------------------------------
    # PRIVADOS: optimización de hiperparámetros
    # --------------------------------------------
    def _optimize_random_forest(self,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                n_trials: int) -> Dict[str, Any]:
        print("🔍 Optimizando Random Forest...")
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            if self.task_type == 'regression':
                model = RandomForestRegressor(**params, random_state=self.random_state, n_jobs=-1)
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                score = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2').mean()
            else:
                model = RandomForestClassifier(**params, random_state=self.random_state, n_jobs=-1)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
            return score

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params

    def _optimize_xgboost(self,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          n_trials: int) -> Dict[str, Any]:
        print("🔍 Optimizando XGBoost...")
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10)
            }
            cls = xgb.XGBRegressor if self.task_type == 'regression' else xgb.XGBClassifier
            model = cls(**params, random_state=self.random_state, n_jobs=-1)
            cv = (KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                  if self.task_type == 'regression'
                  else StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state))
            scoring = 'r2' if self.task_type == 'regression' else 'accuracy'
            score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
            return score

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params

    def _optimize_lightgbm(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           n_trials: int) -> Dict[str, Any]:
        print("🔍 Optimizando LightGBM...")
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100)
            }
            cls = lgb.LGBMRegressor if self.task_type == 'regression' else lgb.LGBMClassifier
            model = cls(**params, random_state=self.random_state, n_jobs=-1, verbose=-1)
            cv = (KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                  if self.task_type == 'regression'
                  else StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state))
            scoring = 'r2' if self.task_type == 'regression' else 'accuracy'
            score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
            return score

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params


# Clase de utilidad para ensemble personalizado
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
        for model in self.models.values():
            model.fit(X, y)
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
