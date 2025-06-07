# ── main_optimized_training.py ────────────────────────────────────────────────
# Módulo principal para entrenamiento optimizado de modelos de ventas

import os
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Imports de la pipeline
from ml_pipeline.feature_engineering import AdvancedFeatureEngineer
from ml_pipeline.data_preprocessing import DataPreprocessor
from ml_pipeline.model_ensembles import ModelEnsembleFactory
from ml_pipeline.hyperopt_module import HyperparameterOptimizer
from ml_pipeline.evaluation_metrics import ModelEvaluator

warnings.filterwarnings('ignore')

# Configurar logging con salida UTF-8
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = [
    handler,
    logging.FileHandler('model_training.log', encoding='utf-8')
]


class OptimizedSalesPredictor:
    def __init__(self, base_dir: str, config: dict = None):
        self.base_dir = Path(base_dir)
        self.csv_path = self.base_dir / 'data' / 'stores_sales_forecasting.csv'
        self.out_dir = self.base_dir / 'backend'
        self.out_dir.mkdir(exist_ok=True)

        self.config = {
            'feature_engineering': True,
            'data_preprocessing': True,
            'model_ensemble': True,
            'hyperparameter_tuning': True,
            'temporal_modeling': False,
            'external_data': False,
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'random_state': 42,
            'n_jobs': -1
        }
        if config:
            self.config.update(config)

        self.feature_engineer = AdvancedFeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.ensemble_factory = ModelEnsembleFactory()
        self.hyperopt = HyperparameterOptimizer()
        self.evaluator = ModelEvaluator()

        self.models = {}
        self.feature_columns = {}
        self.metrics_history = {}

    def load_and_prepare_data(self) -> pd.DataFrame:
        logger.info(f"Cargando datos desde {self.csv_path}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No se encontró el CSV en: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path, encoding='latin1')
        logger.info(f"Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")

        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'], errors='coerce')
        self.df['Year'] = self.df['Order Date'].dt.year
        self.df['Month'] = self.df['Order Date'].dt.month
        self.df['Day'] = self.df['Order Date'].dt.day

        essential = [
            'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name',
            'Year', 'Month', 'Day', 'Order Date', 'Quantity', 'Profit'
        ]
        self.df = (
            self.df[essential]
            .dropna(subset=['Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name'])
            .reset_index(drop=True)
        )
        logger.info(f"Datos filtrados: {self.df.shape[0]} filas válidas")
        return self.df

    def apply_optimizations(self, target_col: str) -> pd.DataFrame:
        logger.info(f"Optimización para objetivo: {target_col}")
        df_opt = self.df.copy()
        if self.config['feature_engineering']:
            df_opt = self.feature_engineer.create_advanced_features(df_opt)
        if self.config['external_data']:
            df_opt = self.external_integrator.add_external_features(df_opt)
        if self.config['temporal_modeling']:
            df_opt = self.temporal_engine.create_temporal_features(df_opt, target_col)
        if self.config['data_preprocessing']:
            df_opt = self.preprocessor.full_preprocessing(df_opt, target_col)
        return df_opt

    def train_optimized_model(self, target_col: str):
        logger.info(f"Entrenamiento optimizado para {target_col}")
        start = datetime.now()

        df_opt = self.apply_optimizations(target_col)
        y = df_opt[target_col]
        X = df_opt.drop(columns=[target_col, 'Order Date'])
        X = pd.get_dummies(X, drop_first=True)
        self.feature_columns[target_col] = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )

        if self.config['data_preprocessing']:
            logger.info("🔄 Preparando datos para escalado...")
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)
            X_train, X_test = self.preprocessor.scale_features(X_train, X_test)

        # Guardamos nombres de columnas antes de transformar a numpy
        if self.config['model_ensemble']:
            self.ensemble_factory.set_feature_names(X_train.columns.tolist())

        # Entrenamiento del ensemble
        if self.config['model_ensemble']:
            logger.info("🤖 Entrenando ensemble de modelos...")
            model = self.ensemble_factory.create_optimized_ensemble(
                X_train, y_train
            )
        else:
            logger.info("🌳 Entrenando XGBRegressor individual...")
            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs']
            )

        if self.config['hyperparameter_tuning']:
            model = self.hyperopt.optimize_model(model, X_train, y_train, target_col)

        X_fit = X_train.values if hasattr(X_train, 'values') else X_train
        y_fit = y_train.values if hasattr(y_train, 'values') else y_train
        model.fit(X_fit, y_fit)

        logger.info("Modelo entrenado, evaluando...")
        X_tr_eval = X_train
        X_te_eval = X_test
        metrics = self.evaluator.comprehensive_evaluation(
            model,
            X_tr_eval.values, X_te_eval.values,
            y_train.values, y_test.values,
            feature_names=self.feature_columns[target_col]
        )

        self.models[target_col] = model
        self.metrics_history[target_col] = metrics

        if self.config['model_ensemble']:
            ensemble_results = self.ensemble_factory.evaluate_ensemble_performance(
                model, X_test, y_test
            )
            self.ensemble_factory.save_evaluation_results(
                ensemble_results,
                str(self.out_dir / f'ensemble_{target_col}_results.csv')
            )

        duration = datetime.now() - start
        logger.info(f"Entrenamiento {target_col} finalizado en {duration}")
        return model, metrics

    def save_models(self):
        logger.info("Guardando modelos y artefactos...")
        for col, m in self.models.items():
            joblib.dump(m, str(self.out_dir / f"model_{col}.pkl"))
            joblib.dump(self.feature_columns[col], str(self.out_dir / f"features_{col}.pkl"))
            joblib.dump(self.metrics_history[col], str(self.out_dir / f"metrics_{col}.pkl"))
        joblib.dump(self.config, str(self.out_dir / "config.pkl"))
        logger.info("Todos los archivos guardados en %s", self.out_dir)

    def generate_report(self):
        report = {"timestamp": datetime.now().isoformat(), "metrics": self.metrics_history}
        logger.info("Reporte generado")
        return report


def main():
    BASE_DIR = r"D:\Repositorios\Modelos_ML"
    config = {
        'feature_engineering': True,
        'data_preprocessing': True,
        'model_ensemble': True,
        'hyperparameter_tuning': True,
        'temporal_modeling': False,
        'external_data': False,
        'cross_validation_folds': 5,
        'test_size': 0.2,
        'random_state': 42,
        'n_jobs': -1
    }
    predictor = OptimizedSalesPredictor(BASE_DIR, config)
    try:
        predictor.load_and_prepare_data()
        predictor.train_optimized_model('Quantity')
        predictor.train_optimized_model('Profit')
        predictor.save_models()
        predictor.generate_report()
        logger.info("Optimización completada exitosamente")
    except Exception as e:
        logger.error("Error en pipeline: %s", e)
        raise


if __name__ == "__main__":
    main()
