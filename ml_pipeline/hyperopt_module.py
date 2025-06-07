# ── hyperparameter_optimizer.py ──────────────────────────────────────────────
# Módulo para optimización de hiperparámetros usando múltiples estrategias
# Autor: Claude AI - Optimización avanzada de modelos
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Imports para optimización
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Imports de modelos
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Optimizador avanzado de hiperparámetros usando múltiples estrategias:
    - Grid Search para espacios pequeños
    - Random Search para espacios medianos  
    - Optuna para optimización bayesiana avanzada
    """
    
    def __init__(self, optimization_method: str = 'optuna', cv_folds: int = 5, 
                 n_trials: int = 100, timeout: int = 1800):
        """
        Inicializa el optimizador
        
        Args:
            optimization_method: 'grid', 'random', 'optuna'
            cv_folds: Número de folds para cross-validation
            n_trials: Número de trials para Optuna
            timeout: Timeout en segundos para Optuna
        """
        self.optimization_method = optimization_method
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.timeout = timeout
        
        # Scorer personalizado que prioriza R²
        self.scorer = make_scorer(r2_score, greater_is_better=True)
        
        # Historial de optimizaciones
        self.optimization_history = {}
        
    def get_parameter_space(self, model_type: str, target_col: str) -> Dict[str, Any]:
        """
        Define espacios de búsqueda optimizados para cada modelo
        
        Args:
            model_type: Tipo de modelo ('xgb', 'lgbm', 'rf', etc.)
            target_col: 'Quantity' o 'Profit' para ajustes específicos
        """
        
        # Parámetros base para XGBoost
        if model_type == 'xgb':
            if self.optimization_method == 'grid':
                return {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [6, 8, 10],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9]
                }
            elif self.optimization_method == 'random':
                return {
                    'n_estimators': [100, 200, 300, 500, 800],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 6, 8, 10, 12],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.01, 0.1, 1],
                    'reg_lambda': [0, 0.01, 0.1, 1]
                }
        
        # Parámetros para LightGBM
        elif model_type == 'lgbm':
            if self.optimization_method == 'grid':
                return {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [6, 8, 10],
                    'num_leaves': [31, 50, 100]
                }
            elif self.optimization_method == 'random':
                return {
                    'n_estimators': [100, 200, 300, 500, 800],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 6, 8, 10, 12],
                    'num_leaves': [15, 31, 50, 100, 200],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
                }
        
        # Parámetros para Random Forest
        elif model_type == 'rf':
            return {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8]
            }
        
        # Parámetros por defecto
        return {}
    
    def optimize_with_optuna(self, model, X_train, y_train, target_col: str):
        """
        Optimización usando Optuna (Bayesian Optimization)
        """
        logger.info(f"🎯 Iniciando optimización Optuna para {target_col}")
        
        # Determinar tipo de modelo
        model_name = model.__class__.__name__.lower()
        if 'xgb' in model_name:
            model_type = 'xgb'
        elif 'lgbm' in model_name or 'lightgbm' in model_name:
            model_type = 'lgbm'
        elif 'randomforest' in model_name:
            model_type = 'rf'
        else:
            model_type = 'xgb'  # Por defecto
        
        def objective(trial):
            """Función objetivo para Optuna"""
            
            # Sugerir parámetros según el modelo
            if model_type == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                }
                
                # Crear modelo con parámetros sugeridos
                opt_model = XGBRegressor(
                    random_state=42,
                    n_jobs=-1,
                    **params
                )
                
            elif model_type == 'lgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
                }
                
                opt_model = LGBMRegressor(
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    **params
                )
            
            else:  # Random Forest o por defecto
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8])
                }
                
                opt_model = RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1,
                    **params
                )
            
            # Cross-validation
            try:
                scores = cross_val_score(
                    opt_model, X_train, y_train,
                    cv=self.cv_folds,
                    scoring=self.scorer,
                    n_jobs=-1
                )
                return scores.mean()
            except Exception as e:
                logger.warning(f"Error en trial {trial.number}: {str(e)}")
                return -999  # Penalizar trials que fallan
        
        # Crear y ejecutar estudio Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(
            objective, 
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Obtener mejores parámetros
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"✅ Mejor R² encontrado: {best_score:.4f}")
        logger.info(f"🔧 Mejores parámetros: {best_params}")
        
        # Crear modelo final con mejores parámetros
        if model_type == 'xgb':
            optimized_model = XGBRegressor(random_state=42, n_jobs=-1, **best_params)
        elif model_type == 'lgbm':
            optimized_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **best_params)
        else:
            optimized_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
        
        # Guardar historial
        self.optimization_history[target_col] = {
            'method': 'optuna',
            'best_score': best_score,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'study': study
        }
        
        return optimized_model
    
    def optimize_with_gridsearch(self, model, X_train, y_train, target_col: str):
        """Optimización usando Grid Search"""
        logger.info(f"🔍 Iniciando Grid Search para {target_col}")
        
        # Obtener espacio de parámetros
        model_type = self._get_model_type(model)
        param_space = self.get_parameter_space(model_type, target_col)
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_space,
            cv=self.cv_folds,
            scoring=self.scorer,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"✅ Mejor R²: {grid_search.best_score_:.4f}")
        logger.info(f"🔧 Mejores parámetros: {grid_search.best_params_}")
        
        # Guardar historial
        self.optimization_history[target_col] = {
            'method': 'grid',
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }
        
        return grid_search.best_estimator_
    
    def optimize_with_randomsearch(self, model, X_train, y_train, target_col: str):
        """Optimización usando Random Search"""
        logger.info(f"🎲 Iniciando Random Search para {target_col}")
        
        # Obtener espacio de parámetros
        model_type = self._get_model_type(model)
        param_space = self.get_parameter_space(model_type, target_col)
        
        # Random Search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_space,
            n_iter=min(50, self.n_trials),
            cv=self.cv_folds,
            scoring=self.scorer,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"✅ Mejor R²: {random_search.best_score_:.4f}")
        logger.info(f"🔧 Mejores parámetros: {random_search.best_params_}")
        
        # Guardar historial
        self.optimization_history[target_col] = {
            'method': 'random',
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'cv_results': random_search.cv_results_
        }
        
        return random_search.best_estimator_
    
    def _get_model_type(self, model) -> str:
        """Determina el tipo de modelo"""
        model_name = model.__class__.__name__.lower()
        
        if 'xgb' in model_name:
            return 'xgb'
        elif 'lgbm' in model_name or 'lightgbm' in model_name:
            return 'lgbm'
        elif 'randomforest' in model_name:
            return 'rf'
        elif 'gradientboosting' in model_name:
            return 'gb'
        else:
            return 'xgb'  # Por defecto
    
    def optimize_model(self, model, X_train, y_train, target_col: str):
        """
        Método principal para optimizar un modelo
        
        Args:
            model: Modelo a optimizar
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            target_col: Nombre de la columna objetivo
            
        Returns:
            Modelo optimizado
        """
        logger.info(f"🚀 Iniciando optimización de hiperparámetros para {target_col}")
        logger.info(f"📊 Método seleccionado: {self.optimization_method}")
        logger.info(f"📈 Datos de entrenamiento: {X_train.shape}")
        
        try:
            # Seleccionar método de optimización
            if self.optimization_method == 'optuna':
                optimized_model = self.optimize_with_optuna(model, X_train, y_train, target_col)
            elif self.optimization_method == 'grid':
                optimized_model = self.optimize_with_gridsearch(model, X_train, y_train, target_col)
            elif self.optimization_method == 'random':
                optimized_model = self.optimize_with_randomsearch(model, X_train, y_train, target_col)
            else:
                logger.warning(f"Método {self.optimization_method} no reconocido. Usando Optuna.")
                optimized_model = self.optimize_with_optuna(model, X_train, y_train, target_col)
            
            logger.info(f"✅ Optimización completada para {target_col}")
            return optimized_model
            
        except Exception as e:
            logger.error(f"❌ Error durante optimización: {str(e)}")
            logger.info("🔄 Devolviendo modelo original")
            return model
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Retorna resumen de todas las optimizaciones realizadas"""
        return {
            'method_used': self.optimization_method,
            'cv_folds': self.cv_folds,
            'optimization_history': self.optimization_history,
            'models_optimized': list(self.optimization_history.keys())
        }
    
    def compare_optimization_methods(self, model, X_train, y_train, target_col: str, 
                                   methods: list = ['grid', 'random', 'optuna']):
        """
        Compara diferentes métodos de optimización
        
        Args:
            model: Modelo base
            X_train, y_train: Datos de entrenamiento
            target_col: Columna objetivo
            methods: Lista de métodos a comparar
            
        Returns:
            Diccionario con resultados de comparación
        """
        logger.info(f"🔄 Comparando métodos de optimización para {target_col}")
        
        results = {}
        original_method = self.optimization_method
        
        for method in methods:
            logger.info(f"🧪 Probando método: {method}")
            self.optimization_method = method
            
            try:
                optimized_model = self.optimize_model(model, X_train, y_train, f"{target_col}_{method}")
                
                # Evaluar con cross-validation
                scores = cross_val_score(
                    optimized_model, X_train, y_train,
                    cv=self.cv_folds, scoring=self.scorer, n_jobs=-1
                )
                
                results[method] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'model': optimized_model
                }
                
            except Exception as e:
                logger.error(f"Error con método {method}: {str(e)}")
                results[method] = {'error': str(e)}
        
        # Restaurar método original
        self.optimization_method = original_method
        
        # Encontrar mejor método
        best_method = max(results.keys(), 
                         key=lambda x: results[x].get('mean_score', -999))
        
        logger.info(f"🏆 Mejor método: {best_method} (R² = {results[best_method]['mean_score']:.4f})")
        
        return {
            'results': results,
            'best_method': best_method,
            'best_model': results[best_method].get('model')
        }
