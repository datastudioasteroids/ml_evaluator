# ── feature_engineering.py ───────────────────────────────────────────────────
# Módulo de ingeniería de features avanzada para mejorar R² sustancialmente
# Incluye features temporales, agregaciones, interacciones y lag features
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Clase para crear features avanzadas que mejoren significativamente el R²
    Implementa múltiples técnicas de feature engineering
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.aggregation_cache = {}
        
    def create_advanced_features(self, df):
        """
        Crea el conjunto completo de features avanzadas
        
        Args:
            df: DataFrame con datos base
            
        Returns:
            DataFrame con features avanzadas añadidas
        """
        logger.info("🔧 Creando features avanzadas...")
        
        df = df.copy()
        
        # 1. Features temporales avanzadas
        df = self._create_temporal_features(df)
        
        # 2. Features de agregación
        df = self._create_aggregation_features(df)
        
        # 3. Features de interacción
        df = self._create_interaction_features(df)
        
        # 4. Lag features y rolling statistics
        df = self._create_lag_features(df)
        
        # 5. Features estadísticas
        df = self._create_statistical_features(df)
        
        # 6. Features de tendencia
        df = self._create_trend_features(df)
        
        # 7. Features de estacionalidad
        df = self._create_seasonal_features(df)
        
        logger.info(f"✅ Features avanzadas creadas. Nuevas dimensiones: {df.shape[1]}")
        return df
    
    def _create_temporal_features(self, df):
        """Crea features temporales avanzadas"""
        logger.info("📅 Creando features temporales...")
        
        # Features básicas adicionales
        df['Quarter'] = df['Order Date'].dt.quarter
        df['DayOfWeek'] = df['Order Date'].dt.dayofweek
        df['WeekOfYear'] = df['Order Date'].dt.isocalendar().week
        df['DayOfYear'] = df['Order Date'].dt.dayofyear
        
        # Features de contexto temporal
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsMonthStart'] = df['Order Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Order Date'].dt.is_month_end.astype(int)
        df['IsQuarterStart'] = df['Order Date'].dt.is_quarter_start.astype(int)
        df['IsQuarterEnd'] = df['Order Date'].dt.is_quarter_end.astype(int)
        
        # Features de tiempo transcurrido
        df['DaysFromStart'] = (df['Order Date'] - df['Order Date'].min()).dt.days
        df['DaysFromEnd'] = (df['Order Date'].max() - df['Order Date']).dt.days
        
        # Features cíclicas (importante para capturar patrones)
        df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeekSin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeekCos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfYearSin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYearCos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        return df
    
    def _create_aggregation_features(self, df):
        """Crea features de agregación por diferentes grupos"""
        logger.info("📊 Creando features de agregación...")
        
        # Definir columnas numéricas para agregación
        numeric_cols = ['Quantity', 'Profit']
        
        # Grupos para agregación
        groupby_cols = [
            'Region',
            'Category', 
            'Sub-Category',
            'Product ID',
            'Year',
            'Month',
            'Quarter',
            'DayOfWeek'
        ]
        
        # Crear agregaciones
        for group_col in groupby_cols:
            if group_col in df.columns:
                for target_col in numeric_cols:
                    if target_col in df.columns:
                        # Estadísticas básicas
                        df[f'{target_col}_{group_col}_Mean'] = df.groupby(group_col)[target_col].transform('mean')
                        df[f'{target_col}_{group_col}_Std'] = df.groupby(group_col)[target_col].transform('std')
                        df[f'{target_col}_{group_col}_Median'] = df.groupby(group_col)[target_col].transform('median')
                        df[f'{target_col}_{group_col}_Min'] = df.groupby(group_col)[target_col].transform('min')
                        df[f'{target_col}_{group_col}_Max'] = df.groupby(group_col)[target_col].transform('max')
                        
                        # Conteos
                        df[f'{group_col}_Count'] = df.groupby(group_col)[target_col].transform('count')
                        
                        # Percentiles
                        df[f'{target_col}_{group_col}_Q25'] = df.groupby(group_col)[target_col].transform(lambda x: x.quantile(0.25))
                        df[f'{target_col}_{group_col}_Q75'] = df.groupby(group_col)[target_col].transform(lambda x: x.quantile(0.75))
        
        return df
    
    def _create_interaction_features(self, df):
        """Crea features de interacción entre variables categóricas"""
        logger.info("🔄 Creando features de interacción...")
        
        # Interacciones importantes
        df['Region_Category'] = df['Region'] + '_' + df['Category']
        df['Region_SubCategory'] = df['Region'] + '_' + df['Sub-Category']
        df['Category_SubCategory'] = df['Category'] + '_' + df['Sub-Category']
        df['Category_Month'] = df['Category'] + '_' + df['Month'].astype(str)
        df['Region_Month'] = df['Region'] + '_' + df['Month'].astype(str)
        df['Category_Quarter'] = df['Category'] + '_' + df['Quarter'].astype(str)
        df['Region_Quarter'] = df['Region'] + '_' + df['Quarter'].astype(str)
        df['Category_DayOfWeek'] = df['Category'] + '_' + df['DayOfWeek'].astype(str)
        
        # Interacciones temporales más complejas
        df['YearMonth'] = df['Year'].astype(str) + '_' + df['Month'].astype(str)
        df['YearQuarter'] = df['Year'].astype(str) + '_Q' + df['Quarter'].astype(str)
        
        return df
    
    def _create_lag_features(self, df):
        """Crea lag features y rolling statistics"""
        logger.info("⏳ Creando lag features y rolling statistics...")
        
        # Ordenar por fecha para lags
        df = df.sort_values('Order Date').reset_index(drop=True)
        
        # Crear lags por producto
        for target_col in ['Quantity', 'Profit']:
            if target_col in df.columns:
                # Lags por producto
                df[f'{target_col}_Lag1'] = df.groupby('Product ID')[target_col].shift(1)
                df[f'{target_col}_Lag3'] = df.groupby('Product ID')[target_col].shift(3)
                df[f'{target_col}_Lag7'] = df.groupby('Product ID')[target_col].shift(7)
                
                # Rolling statistics por producto
                df[f'{target_col}_RollingMean_3'] = df.groupby('Product ID')[target_col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{target_col}_RollingMean_7'] = df.groupby('Product ID')[target_col].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{target_col}_RollingStd_3'] = df.groupby('Product ID')[target_col].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
                df[f'{target_col}_RollingStd_7'] = df.groupby('Product ID')[target_col].rolling(window=7, min_periods=1).std().reset_index(0, drop=True)
                
                # Rolling por categoría
                df[f'{target_col}_Category_RollingMean_7'] = df.groupby('Category')[target_col].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{target_col}_Region_RollingMean_7'] = df.groupby('Region')[target_col].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        
        # Rellenar NaN con forward fill
        lag_cols = [col for col in df.columns if 'Lag' in col or 'Rolling' in col]
        for col in lag_cols:
            df[col] = df[col].fillna(method='ffill').fillna(0)
        
        return df
    
    def _create_statistical_features(self, df):
        """Crea features estadísticas avanzadas"""
        logger.info("📈 Creando features estadísticas...")
        
        for target_col in ['Quantity', 'Profit']:
            if target_col in df.columns:
                # Z-scores
                df[f'{target_col}_ZScore'] = (df[target_col] - df[target_col].mean()) / df[target_col].std()
                
                # Outlier indicators
                Q1 = df[target_col].quantile(0.25)
                Q3 = df[target_col].quantile(0.75)
                IQR = Q3 - Q1
                df[f'{target_col}_IsOutlier'] = ((df[target_col] < (Q1 - 1.5 * IQR)) | 
                                                (df[target_col] > (Q3 + 1.5 * IQR))).astype(int)
                
                # Percentile ranks
                df[f'{target_col}_PercentileRank'] = df[target_col].rank(pct=True)
                
                # Log transformations (para datos skewed)
                df[f'{target_col}_Log'] = np.log1p(np.abs(df[target_col]))
                
                # Square root transformation
                df[f'{target_col}_Sqrt'] = np.sqrt(np.abs(df[target_col]))
        
        return df
    
    def _create_trend_features(self, df):
        """Crea features de tendencia temporal"""
        logger.info("📊 Creando features de tendencia...")
        
        # Ordenar por fecha
        df = df.sort_values('Order Date').reset_index(drop=True)
        
        for target_col in ['Quantity', 'Profit']:
            if target_col in df.columns:
                # Diferencias con períodos anteriores
                df[f'{target_col}_Diff1'] = df[target_col].diff(1)
                df[f'{target_col}_Diff7'] = df[target_col].diff(7)
                df[f'{target_col}_Diff30'] = df[target_col].diff(30)
                
                # Ratios con períodos anteriores
                df[f'{target_col}_Ratio1'] = df[target_col] / (df[target_col].shift(1) + 1e-8)
                df[f'{target_col}_Ratio7'] = df[target_col] / (df[target_col].shift(7) + 1e-8)
                
                # Momentum indicators
                df[f'{target_col}_Momentum_3'] = df[target_col] / (df[target_col].shift(3) + 1e-8)
                df[f'{target_col}_Momentum_7'] = df[target_col] / (df[target_col].shift(7) + 1e-8)
                
                # Volatility (rolling std / rolling mean)
                rolling_mean = df[target_col].rolling(window=7, min_periods=1).mean()
                rolling_std = df[target_col].rolling(window=7, min_periods=1).std()
                df[f'{target_col}_Volatility'] = rolling_std / (rolling_mean + 1e-8)
        
        # Rellenar NaN
        trend_cols = [col for col in df.columns if any(x in col for x in ['Diff', 'Ratio', 'Momentum', 'Volatility'])]
        for col in trend_cols:
            df[col] = df[col].fillna(method='ffill').fillna(0)
        
        return df
    
    def _create_seasonal_features(self, df):
        """Crea features de estacionalidad avanzada"""
        logger.info("🌍 Creando features de estacionalidad...")
        
        # Detectar patrones estacionales
        for target_col in ['Quantity', 'Profit']:
            if target_col in df.columns:
                # Promedio por mes histórico
                monthly_avg = df.groupby('Month')[target_col].mean()
                df[f'{target_col}_MonthlyHistoricalAvg'] = df['Month'].map(monthly_avg)
                
                # Promedio por día de la semana
                dow_avg = df.groupby('DayOfWeek')[target_col].mean()
                df[f'{target_col}_DowHistoricalAvg'] = df['DayOfWeek'].map(dow_avg)
                
                # Promedio por quarter
                quarter_avg = df.groupby('Quarter')[target_col].mean()
                df[f'{target_col}_QuarterHistoricalAvg'] = df['Quarter'].map(quarter_avg)
                
                # Desviación de la norma estacional
                df[f'{target_col}_SeasonalDeviation_Month'] = df[target_col] - df[f'{target_col}_MonthlyHistoricalAvg']
                df[f'{target_col}_SeasonalDeviation_Quarter'] = df[target_col] - df[f'{target_col}_QuarterHistoricalAvg']
                
                # Índice estacional
                overall_avg = df[target_col].mean()
                df[f'{target_col}_SeasonalIndex_Month'] = df[f'{target_col}_MonthlyHistoricalAvg'] / overall_avg
                df[f'{target_col}_SeasonalIndex_Quarter'] = df[f'{target_col}_QuarterHistoricalAvg'] / overall_avg
        
        return df
    
    def create_performance_indicators(self, df):
        """
        Crea indicadores de rendimiento específicos para el negocio
        
        Args:
            df: DataFrame con datos procesados
            
        Returns:
            DataFrame con indicadores de rendimiento
        """
        logger.info("📊 Creando indicadores de rendimiento...")
        
        df = df.copy()
        
        # Indicadores de eficiencia
        if 'Sales' in df.columns and 'Profit' in df.columns:
            df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
            df['Profit_Margin_Category'] = pd.cut(df['Profit_Margin'], 
                                                 bins=[-np.inf, 0, 10, 20, np.inf],
                                                 labels=['Pérdida', 'Bajo', 'Medio', 'Alto'])
        
        # Indicadores de volumen
        if 'Quantity' in df.columns:
            df['Quantity_Percentile'] = df['Quantity'].rank(pct=True)
            df['High_Volume'] = (df['Quantity'] > df['Quantity'].quantile(0.8)).astype(int)
        
        # Indicadores de cliente/región
        if 'Customer ID' in df.columns:
            customer_stats = df.groupby('Customer ID').agg({
                'Sales': ['count', 'sum', 'mean'],
                'Profit': 'sum'
            }).round(2)
            customer_stats.columns = ['Order_Count', 'Total_Sales', 'Avg_Order_Value', 'Total_Profit']
            df = df.merge(customer_stats, left_on='Customer ID', right_index=True, how='left')
        
        # Indicadores temporales de rendimiento
        df['Weekend_Performance'] = np.where(df['IsWeekend'], 'Weekend', 'Weekday')
        df['Month_Performance_Tier'] = pd.cut(df['Month'], 
                                            bins=[0, 3, 6, 9, 12],
                                            labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        logger.info("✅ Indicadores de rendimiento creados")
        return df
    
    def encode_categorical_features(self, df):
        """
        Codifica features categóricas para modelos ML
        
        Args:
            df: DataFrame con features categóricas
            
        Returns:
            DataFrame con features codificadas
        """
        logger.info("🔢 Codificando features categóricas...")
        
        df = df.copy()
        
        # Identificar columnas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Excluir columnas de fecha si existen
        categorical_cols = [col for col in categorical_cols if 'Date' not in col]
        
        # Codificar con Label Encoder
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_Encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Para datos nuevos, usar el encoder ya entrenado
                try:
                    df[col + '_Encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Si hay nuevas categorías, usar -1
                    df[col + '_Encoded'] = df[col].map({v: k for k, v in enumerate(self.label_encoders[col].classes_)}).fillna(-1)
        
        logger.info("✅ Features categóricas codificadas")
        return df
    
    def get_feature_importance_report(self, df, target_col):
        """
        Genera un reporte de importancia de features
        
        Args:
            df: DataFrame con features
            target_col: Columna objetivo
            
        Returns:
            DataFrame con importancia de features
        """
        logger.info("📈 Generando reporte de importancia de features...")
        
        
        
        # Preparar datos
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        X = df[numeric_cols].fillna(0)
        y = df[target_col]
        
        # Random Forest Feature Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': numeric_cols,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        # Mutual Information
        mi_scores = mutual_info_regression(X, y)
        mi_importance = pd.DataFrame({
            'feature': numeric_cols,
            'mi_importance': mi_scores
        }).sort_values('mi_importance', ascending=False)
        
        # Combinar resultados
        importance_report = rf_importance.merge(mi_importance, on='feature')
        importance_report['combined_score'] = (importance_report['rf_importance'] + 
                                             importance_report['mi_importance']) / 2
        importance_report = importance_report.sort_values('combined_score', ascending=False)
        
        logger.info("✅ Reporte de importancia generado")
        return importance_report
    
    def clean_and_validate_features(self, df):
        """
        Limpia y valida el dataset final
        
        Args:
            df: DataFrame con todas las features
            
        Returns:
            DataFrame limpio y validado
        """
        logger.info("🧹 Limpiando y validando features...")
        
        df = df.copy()
        
        # Remover columnas con demasiados NaN
        threshold = 0.5  # 50% de valores faltantes
        nan_ratio = df.isnull().sum() / len(df)
        cols_to_drop = nan_ratio[nan_ratio > threshold].index.tolist()
        
        if cols_to_drop:
            logger.warning(f"Removiendo columnas con >50% NaN: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Rellenar NaN restantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        # Remover columnas con varianza cero
        numeric_df = df.select_dtypes(include=[np.number])
        zero_var_cols = numeric_df.columns[numeric_df.var() == 0].tolist()
        
        if zero_var_cols:
            logger.warning(f"Removiendo columnas con varianza cero: {zero_var_cols}")
            df = df.drop(columns=zero_var_cols)
        
        # Validar infinitos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        
        logger.info(f"✅ Dataset limpio. Dimensiones finales: {df.shape}")
        return df


# Función principal de ejecución
def main():
    """Función principal para demostrar el uso de AdvancedFeatureEngineer"""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("🚀 Iniciando Advanced Feature Engineering...")
    
    # Ejemplo de uso
    # df = pd.read_csv('your_data.csv')
    # df['Order Date'] = pd.to_datetime(df['Order Date'])
    
    # Inicializar el ingeniero de features
    # engineer = AdvancedFeatureEngineer()
    
    # Crear features avanzadas
    # df_advanced = engineer.create_advanced_features(df)
    # df_with_kpis = engineer.create_performance_indicators(df_advanced)
    # df_encoded = engineer.encode_categorical_features(df_with_kpis)
    # df_final = engineer.clean_and_validate_features(df_encoded)
    
    # Generar reporte de importancia
    # importance_report = engineer.get_feature_importance_report(df_final, 'target_column')
    
    logger.info("✅ Proceso completado exitosamente!")


if __name__ == "__main__":
    main()