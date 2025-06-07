# ── data_preprocessing.py ────────────────────────────────────────────────────
# Módulo de preprocesamiento avanzado de datos para optimizar R²
# Incluye limpieza, transformaciones, escalado y validación de datos
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import zscore
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Clase para preprocesamiento avanzado de datos
    Implementa múltiples técnicas de limpieza y transformación
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.outlier_bounds = {}
        
    def full_preprocessing(self, df, target_col):
        """
        Ejecuta el pipeline completo de preprocesamiento
        
        Args:
            df: DataFrame con datos brutos
            target_col: Columna objetivo ('Quantity' o 'Profit')
            
        Returns:
            DataFrame preprocesado y optimizado
        """
        logger.info(f"🔧 Iniciando preprocesamiento completo para {target_col}...")
        
        df = df.copy()
        
        # 1. Limpieza básica de datos
        df = self._basic_data_cleaning(df)
        
        # 2. Manejo de outliers
        df = self._handle_outliers(df, target_col)
        
        # 3. Imputación de valores faltantes
        df = self._impute_missing_values(df)
        
        # 4. Transformaciones de variables
        df = self._apply_transformations(df, target_col)
        
        # 5. Encoding de variables categóricas
        df = self._encode_categorical_variables(df)
        
        # 6. Selección de features
        df = self._feature_selection(df, target_col)
        
        # 7. Validación final
        df = self._final_validation(df)
        
        logger.info(f"✅ Preprocesamiento completado. Dimensiones: {df.shape}")
        return df
    
    def _basic_data_cleaning(self, df):
        """Limpieza básica de datos"""
        logger.info("🧹 Realizando limpieza básica...")
        
        # Eliminar filas completamente vacías
        df = df.dropna(how='all')
        
        # Eliminar columnas con más del 90% de valores faltantes
        threshold = 0.9
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        if cols_to_drop:
            logger.warning(f"Eliminando columnas con >90% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Eliminar duplicados exactos
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        removed_duplicates = initial_shape - df.shape[0]
        
        if removed_duplicates > 0:
            logger.info(f"Eliminados {removed_duplicates} registros duplicados")
        
        # Limpiar strings (espacios, caracteres especiales)
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            if col != 'Order Date':  # Preservar fechas
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', np.nan)
        
        return df
    
    def _handle_outliers(self, df, target_col):
        """Manejo inteligente de outliers"""
        logger.info("📊 Manejando outliers...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                # Calcular bounds usando IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Almacenar bounds para uso posterior
                self.outlier_bounds[col] = {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
                
                # Estrategia diferente para variable objetivo vs features
                if col == target_col:
                    # Para target: solo eliminar outliers extremos (3*IQR)
                    extreme_lower = Q1 - 3 * IQR
                    extreme_upper = Q3 + 3 * IQR
                    outlier_mask = (df[col] < extreme_lower) | (df[col] > extreme_upper)
                    
                    if outlier_mask.sum() > 0:
                        logger.info(f"Eliminando {outlier_mask.sum()} outliers extremos de {col}")
                        df = df[~outlier_mask]
                else:
                    # Para features: winsorizing (cap values)
                    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    
                    if outlier_count > 0:
                        logger.info(f"Aplicando winsorizing a {outlier_count} outliers en {col}")
                        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        return df
    
    def _impute_missing_values(self, df):
        """Imputación avanzada de valores faltantes"""
        logger.info("🔄 Imputando valores faltantes...")
        
        # Separar columnas numéricas y categóricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Imputación numérica con KNN para mejor precisión
        if numeric_cols:
            missing_numeric = df[numeric_cols].isnull().sum()
            cols_with_missing = missing_numeric[missing_numeric > 0].index.tolist()
            
            if cols_with_missing:
                logger.info(f"Imputando columnas numéricas: {cols_with_missing}")
                
                # Usar KNN si no hay demasiados missing, sino media/mediana
                missing_ratio = df[cols_with_missing].isnull().sum() / len(df)
                
                if missing_ratio.max() < 0.3:  # Menos del 30% missing
                    imputer = KNNImputer(n_neighbors=5)
                    self.imputers['numeric_knn'] = imputer
                    df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])
                else:
                    # Usar mediana para robustez
                    imputer = SimpleImputer(strategy='median')
                    self.imputers['numeric_median'] = imputer
                    df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])
        
        # Imputación categórica
        if categorical_cols:
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    # Usar moda o 'Unknown' según el caso
                    if df[col].value_counts().iloc[0] / len(df) > 0.5:  # Moda muy dominante
                        mode_value = df[col].mode().iloc[0]
                        df[col] = df[col].fillna(mode_value)
                    else:
                        df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _apply_transformations(self, df, target_col):
        """Aplica transformaciones matemáticas para mejorar distribuciones"""
        logger.info("🔄 Aplicando transformaciones...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col in df.columns and col != target_col:
                # Detectar skewness
                skewness = stats.skew(df[col].dropna())
                
                if abs(skewness) > 1:  # Distribución muy sesgada
                    logger.info(f"Transformando columna sesgada: {col} (skew: {skewness:.2f})")
                    
                    # Aplicar transformación según el tipo de skew
                    if skewness > 1:  # Positively skewed
                        # Log transformation (agregar constante para valores negativos)
                        if df[col].min() <= 0:
                            df[col + '_log'] = np.log1p(df[col] - df[col].min() + 1)
                        else:
                            df[col + '_log'] = np.log1p(df[col])
                        
                        # Square root transformation
                        if df[col].min() >= 0:
                            df[col + '_sqrt'] = np.sqrt(df[col])
                    
                    elif skewness < -1:  # Negatively skewed
                        # Square transformation
                        df[col + '_square'] = np.square(df[col])
                
                # Box-Cox transformation para mejorar normalidad
                if df[col].min() > 0:  # Box-Cox requiere valores positivos
                    try:
                        transformed, lambda_param = stats.boxcox(df[col])
                        if abs(stats.skew(transformed)) < abs(skewness):
                            df[col + '_boxcox'] = transformed
                    except:
                        pass  # Si falla, continuar sin Box-Cox
        
        return df
    
    def _encode_categorical_variables(self, df):
        """Codificación inteligente de variables categóricas"""
        logger.info("🔢 Codificando variables categóricas...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Excluir fechas
        categorical_cols = [col for col in categorical_cols if 'Date' not in col]
        
        for col in categorical_cols:
            unique_values = df[col].nunique()
            
            if unique_values == 2:
                # Binary encoding para variables binarias
                le = LabelEncoder()
                df[col + '_binary'] = le.fit_transform(df[col].astype(str))
                self.encoders[col + '_binary'] = le
                
            elif unique_values <= 10:
                # One-hot encoding para pocas categorías
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                
                # Label encoding también (puede ser útil para algunos modelos)
                le = LabelEncoder()
                df[col + '_label'] = le.fit_transform(df[col].astype(str))
                self.encoders[col + '_label'] = le
                
            else:
                # Solo label encoding para muchas categorías
                le = LabelEncoder()
                df[col + '_label'] = le.fit_transform(df[col].astype(str))
                self.encoders[col + '_label'] = le
                
                # Target encoding si hay suficientes datos
                if len(df) > 1000:
                    target_mean = df.groupby(col)[df.select_dtypes(include=[np.number]).columns[0]].mean()
                    df[col + '_target_encoded'] = df[col].map(target_mean)
        
        return df
    
    def _feature_selection(self, df, target_col):
        """Selección inteligente de features"""
        logger.info("🎯 Realizando selección de features...")
        
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} no encontrada para selección")
            return df
        
        # Preparar datos para selección
        y = df[target_col]
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
        
        # Eliminar features con varianza muy baja
        variance_threshold = 0.01
        low_variance_cols = X.columns[X.var() < variance_threshold].tolist()
        
        if low_variance_cols:
            logger.info(f"Eliminando features con baja varianza: {len(low_variance_cols)}")
            X = X.drop(columns=low_variance_cols)
            df = df.drop(columns=low_variance_cols)
        
        # Eliminar features altamente correlacionadas
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_threshold = 0.95
        high_corr_cols = [column for column in upper_triangle.columns if 
                         any(upper_triangle[column] > high_corr_threshold)]
        
        if high_corr_cols:
            logger.info(f"Eliminando features altamente correlacionadas: {len(high_corr_cols)}")
            df = df.drop(columns=high_corr_cols)
            X = X.drop(columns=high_corr_cols)
        
        # Selección por importancia estadística
        if len(X.columns) > 50:  # Solo si hay muchas features
            k_best = min(50, len(X.columns))  # Mantener máximo 50 features
            
            selector = SelectKBest(score_func=f_regression, k=k_best)
            X_selected = selector.fit_transform(X.fillna(0), y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Mantener solo las features seleccionadas
            features_to_drop = [col for col in X.columns if col not in selected_features]
            if features_to_drop:
                logger.info(f"Seleccionadas {len(selected_features)} mejores features")
                df = df.drop(columns=features_to_drop)
            
            self.feature_selectors[target_col] = selector
        
        return df
    
    def scale_features(self, X_train, X_test, method='standard'):
        """
        Escala features numéricas
        
        Args:
            X_train: Datos de entrenamiento
            X_test: Datos de prueba
            method: 'standard', 'minmax', o 'robust'
            
        Returns:
            X_train_scaled, X_test_scaled
        """
        logger.info(f"📏 Escalando features con método: {method}")
        
        # Seleccionar scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Método {method} no reconocido, usando StandardScaler")
            scaler = StandardScaler()
        
        # Ajustar y transformar
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Guardar scaler para uso posterior
        self.scalers[method] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def _final_validation(self, df):
        """Validación final del dataset preprocesado"""
        logger.info("✅ Realizando validación final...")
        
        # Verificar tipos de datos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Reemplazar infinitos
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Imputar cualquier NaN residual con 0
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Verificar que no hay columnas completamente vacías
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            logger.warning(f"Eliminando columnas vacías: {empty_cols}")
            df = df.drop(columns=empty_cols)
        
        # Verificar varianza
        zero_var_cols = df[numeric_cols].columns[df[numeric_cols].var() == 0].tolist()
        if zero_var_cols:
            logger.warning(f"Eliminando columnas con varianza cero: {zero_var_cols}")
            df = df.drop(columns=zero_var_cols)
        
        logger.info(f"✅ Validación completada. Dataset final: {df.shape}")
        return df
    
    def get_preprocessing_summary(self):
        """Genera resumen del preprocesamiento aplicado"""
        summary = {
            'scalers_used': list(self.scalers.keys()),
            'encoders_created': len(self.encoders),
            'imputers_used': list(self.imputers.keys()),
            'feature_selectors': list(self.feature_selectors.keys()),
            'outlier_bounds_set': len(self.outlier_bounds)
        }
        
        return summary


# ──────────────────────────────────────────────────────────────────────────────
# COMENTARIOS PARA PRÓXIMOS MÓDULOS A CREAR:
# ──────────────────────────────────────────────────────────────────────────────

"""
PRÓXIMO MÓDULO: model_ensembles.py

class ModelEnsembleFactory:
    Debe crear:
    1. create_optimized_ensemble(target_col):
       - Combinar XGBoost, LightGBM, RandomForest
       - Voting/Stacking regressor
       - Optimizado para cada target (Quantity/Profit)
    
    2. create_individual_models():
       - XGBRegressor con parámetros optimizados
       - LGBMRegressor 
       - RandomForestRegressor
       - Ridge/Lasso para regularización
    
    3. create_stacking_ensemble():
       - Usar modelos base como level-0
       - Meta-learner (LinearRegression/Ridge) como level-1
       - Cross-validation para evitar overfitting
    
    4. get_ensemble_weights():
       - Calcular pesos óptimos para voting
       - Basado en performance individual en validation
"""

def main():
    """Función principal para demostrar el uso del DataPreprocessor"""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("🚀 Iniciando DataPreprocessor...")
    
    # Ejemplo de uso
    # df = pd.read_csv('your_data.csv')
    # preprocessor = DataPreprocessor()
    
    # Pipeline completo
    # df_processed = preprocessor.full_preprocessing(df, 'target_column')
    
    # Escalado separado
    # X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test, 'robust')
    
    # Resumen
    # summary = preprocessor.get_preprocessing_summary()
    
    logger.info("✅ Ejemplo completado!")


if __name__ == "__main__":
    main()