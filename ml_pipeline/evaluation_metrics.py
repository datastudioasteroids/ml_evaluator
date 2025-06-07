import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from typing import Dict, Any, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Imports de métricas
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.inspection import permutation_importance

# Para análisis de residuos
from scipy.stats import jarque_bera, shapiro, anderson, kstest
from statsmodels.stats.diagnostic import het_breuschpagan

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluador completo de modelos con métricas avanzadas y análisis estadísticos
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.evaluation_history = {}
        
    def calculate_basic_metrics(self, y_true, y_pred) -> Dict[str, float]:
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mse': mean_squared_error(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
        if not np.any(y_true == 0):
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        else:
            non_zero = y_true != 0
            metrics['mape'] = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if non_zero.any() else np.inf
        return metrics
    
    def calculate_advanced_metrics(self, y_true, y_pred) -> Dict[str, float]:
        residuals = y_true - y_pred
        metrics = {
            'median_ae': np.median(np.abs(residuals)),
            'max_error': np.max(np.abs(residuals)),
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'mae_p25': np.percentile(np.abs(residuals), 25),
            'mae_p75': np.percentile(np.abs(residuals), 75),
            'mae_p90': np.percentile(np.abs(residuals), 90),
            'mae_p95': np.percentile(np.abs(residuals), 95),
            'residual_skewness': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals),
            'pred_residual_corr': np.corrcoef(y_pred, residuals)[0,1] if len(set(y_pred))>1 else 0
        }
        if np.std(y_true) > 0:
            metrics['normalized_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)
            metrics['cv_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true) if np.mean(y_true)!=0 else np.inf
        return metrics
    
    def perform_residual_analysis(self, y_true, y_pred) -> Dict[str, Any]:
        residuals = y_true - y_pred
        std_res = (residuals - residuals.mean()) / residuals.std()
        analysis = {
            'residual_stats': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'min': residuals.min(),
                'max': residuals.max(),
                'range': residuals.max()-residuals.min()
            }
        }
        try:
            if len(residuals) < 5000:
                s_stat, s_p = shapiro(residuals)
                analysis['normality_shapiro'] = {'statistic': s_stat, 'p_value': s_p, 'is_normal': s_p>0.05}
            jb_stat, jb_p = jarque_bera(residuals)
            analysis['normality_jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_p, 'is_normal': jb_p>0.05}
            ad = anderson(residuals, dist='norm')
            analysis['normality_anderson'] = {'statistic': ad.statistic, 'critical_values': ad.critical_values, 'significance_levels': ad.significance_level}
        except Exception as e:
            logger.warning(f"Error en tests de normalidad: {e}")
            analysis['normality_error'] = str(e)
        try:
            X_test = np.column_stack([y_pred, np.ones(len(y_pred))])
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_test)
            analysis['homoscedasticity_bp'] = {'statistic': bp_stat, 'p_value': bp_p, 'is_homoscedastic': bp_p>0.05}
        except Exception as e:
            logger.warning(f"Error en test de homocedasticidad: {e}")
        q75, q25 = np.percentile(std_res, [75,25])
        iqr = q75-q25
        mask = (std_res<q25-1.5*iqr)|(std_res>q75+1.5*iqr)
        analysis['outliers'] = {'count': mask.sum(), 'percentage': mask.mean()*100, 'indices': np.where(mask)[0].tolist(), 'values': std_res[mask].tolist()}
        return analysis
    
    def cross_validation_analysis(self, model, X, y) -> Dict[str, Any]:
        r2 = cross_val_score(model, X, y, cv=self.cv_folds, scoring='r2', n_jobs=-1)
        mae = -cross_val_score(model, X, y, cv=self.cv_folds, scoring='neg_mean_absolute_error', n_jobs=-1)
        rmse = -cross_val_score(model, X, y, cv=self.cv_folds, scoring='neg_root_mean_squared_error', n_jobs=-1)
        return {
            'r2': {'scores': r2, 'mean': r2.mean(), 'std': r2.std(), 'confidence_interval': np.percentile(r2,[2.5,97.5])},
            'mae': {'scores': mae, 'mean': mae.mean(), 'std': mae.std(), 'confidence_interval': np.percentile(mae,[2.5,97.5])},
            'rmse': {'scores': rmse, 'mean': rmse.mean(), 'std': rmse.std(), 'confidence_interval': np.percentile(rmse,[2.5,97.5])}
        }
    
    def learning_curve_analysis(self, model, X, y, train_sizes=None) -> Dict[str, Any]:
        if train_sizes is None:
            train_sizes = np.linspace(0.1,1.0,10)
        sizes, train_scores, val_scores = learning_curve(model, X, y, train_sizes=train_sizes, cv=self.cv_folds, scoring='r2', n_jobs=-1, random_state=self.random_state)
        return {'train_sizes': sizes, 'train_scores': {'mean': train_scores.mean(axis=1), 'std': train_scores.std(axis=1)}, 'validation_scores': {'mean': val_scores.mean(axis=1), 'std': val_scores.std(axis=1)}, 'final_gap': train_scores.mean(axis=1)[-1]-val_scores.mean(axis=1)[-1], 'is_overfitting': train_scores.mean(axis=1)[-1]-val_scores.mean(axis=1)[-1]>0.1}
    
    def feature_importance_analysis(self, model, X, y, feature_names=None) -> Dict[str, Any]:
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        out = {}
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            out['intrinsic'] = {'importances': imp, 'feature_names': feature_names, 'sorted_indices': np.argsort(imp)[::-1]}
        try:
            perm = permutation_importance(model, X, y, n_repeats=10, random_state=self.random_state, scoring='r2', n_jobs=-1)
            out['permutation'] = {'importances_mean': perm.importances_mean, 'importances_std': perm.importances_std, 'feature_names': feature_names, 'sorted_indices': np.argsort(perm.importances_mean)[::-1]}
        except Exception as e:
            logger.warning(f"Error en permutation importance: {e}")
        return out
    
    def validation_curve_analysis(self, model, X, y, param_name, param_range) -> Dict[str, Any]:
        train_scores, val_scores = validation_curve(model, X, y, param_name=param_name, param_range=param_range, cv=self.cv_folds, scoring='r2', n_jobs=-1)
        means = val_scores.mean(axis=1)
        idx = means.argmax()
        return {'param_name': param_name, 'param_range': param_range, 'train_scores': {'mean': train_scores.mean(axis=1), 'std': train_scores.std(axis=1)}, 'validation_scores': {'mean': means, 'std': val_scores.std(axis=1)}, 'best_param_idx': idx, 'best_param_value': param_range[idx], 'best_score': means[idx]}
    
    def comprehensive_evaluation(self, model, X_train, X_test, y_train, y_test, feature_names=None) -> Dict[str, Any]:
        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)
        evals = {
            'model_info': {'model_type': type(model).__name__, 'train_samples': len(X_train), 'test_samples': len(X_test), 'features': X_train.shape[1]},
            'training_performance': {'basic_metrics': self.calculate_basic_metrics(y_train,y_tr_pred), 'advanced_metrics': self.calculate_advanced_metrics(y_train,y_tr_pred), 'residual_analysis': self.perform_residual_analysis(y_train,y_tr_pred)},
            'test_performance': {'basic_metrics': self.calculate_basic_metrics(y_test,y_te_pred), 'advanced_metrics': self.calculate_advanced_metrics(y_test,y_te_pred), 'residual_analysis': self.perform_residual_analysis(y_test,y_te_pred)},
            'cross_validation': self.cross_validation_analysis(model,X_train,y_train),
            'learning_curves': self.learning_curve_analysis(model,X_train,y_train),
            'feature_importance': self.feature_importance_analysis(model,X_train,y_train,feature_names)
        }
        tr_r2 = evals['training_performance']['basic_metrics']['r2']
        te_r2 = evals['test_performance']['basic_metrics']['r2']
        cv_r2 = evals['cross_validation']['r2']['mean']
        evals['model_diagnosis'] = {'train_test_gap': tr_r2-te_r2, 'cv_test_gap': cv_r2-te_r2, 'is_overfitting': tr_r2-te_r2>0.1, 'is_underfitting': cv_r2<0.7, 'generalization_score': min(te_r2,cv_r2), 'stability_score':1-evals['cross_validation']['r2']['std']}
        name = f"{type(model).__name__}_{len(self.evaluation_history)}"
        self.evaluation_history[name] = evals
        logger.info("✅ Evaluación completa finalizada")
        return evals
    
    def plot_evaluation_results(self, evaluation_results: Dict[str, Any], figsize=(20,15), save_path=None):
        fig, axes = plt.subplots(3,4, figsize=figsize)
        fig.suptitle('Evaluación Completa del Modelo', fontsize=16, fontweight='bold')
        
        # 1. Métricas básicas
        metrics = ['r2','mae','rmse']
        train_vals = [evaluation_results['training_performance']['basic_metrics'][m] for m in metrics]
        test_vals  = [evaluation_results['test_performance']['basic_metrics'][m] for m in metrics]
        x = np.arange(len(metrics)); w=0.35
        axes[0,0].bar(x-w/2, train_vals,w,label='Train'); axes[0,0].bar(x+w/2,test_vals,w,label='Test')
        axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(metrics); axes[0,0].set_title('Train vs Test Básicas'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
        
        # 2. Residuos Train
        stats_tr = evaluation_results['training_performance']['residual_analysis']['residual_stats']
        vals = [stats_tr['mean'],stats_tr['std'],stats_tr['min'],stats_tr['max']]
        labels=['Mean','Std','Min','Max']
        cols=['green' if v>=0 else 'red' for v in vals]
        axes[0,1].bar(labels,vals,color=cols,alpha=0.7); axes[0,1].set_title('Residuos (Train)'); axes[0,1].grid(alpha=0.3)
        
        # 3. CV R2
        r2cv = evaluation_results['cross_validation']['r2']['scores']
        axes[0,2].boxplot([r2cv],labels=['R2 CV']); axes[0,2].set_title('R2 CV'); axes[0,2].grid(alpha=0.3)
        
        # 4. Outliers Test
        out_t = evaluation_results['test_performance']['residual_analysis']['outliers']
        sizes=[100-out_t['percentage'],out_t['percentage']]; labels=['Normal','Outliers']
        axes[0,3].pie(sizes,labels=labels,autopct='%1.1f%%',startangle=90); axes[0,3].set_title('Outliers Test')
        
        # 5. Learning Curves
        lc = evaluation_results['learning_curves']
        ts=lc['train_sizes']; t_m=lc['train_scores']['mean']; t_s=lc['train_scores']['std']; v_m=lc['validation_scores']['mean']; v_s=lc['validation_scores']['std']
        axes[1,0].plot(ts,t_m,'o-'); axes[1,0].fill_between(ts,t_m-t_s,t_m+t_s,alpha=0.1)
        axes[1,0].plot(ts,v_m,'o-'); axes[1,0].fill_between(ts,v_m-v_s,v_m+v_s,alpha=0.1)
        axes[1,0].set_title('Curvas Aprendizaje'); axes[1,0].grid(alpha=0.3)
        
        # 6. Permutation Importances
        if 'permutation' in evaluation_results['feature_importance']:
            perm = evaluation_results['feature_importance']['permutation']
            idx=perm['sorted_indices'][:10]
            axes[1,1].barh(np.arange(len(idx)), perm['importances_mean'][idx]);
            axes[1,1].set_yticks(np.arange(len(idx))); axes[1,1].set_yticklabels([perm['feature_names'][i] for i in idx]); axes[1,1].set_title('Top Features')
        else:
            axes[1,1].text(0.5,0.5,'No disponible',ha='center',va='center',transform=axes[1,1].transAxes)
        
        # 7. Diagnóstico
        diag=evaluation_results['model_diagnosis']; md=['generalization_score','stability_score','train_test_gap']
        vals=[diag['generalization_score'],diag['stability_score'],abs(diag['train_test_gap'])]
        cols=[ 'green' if (v>0.8 if i<2 else v<0.1) else 'red' for i,v in enumerate(vals)]
        axes[1,2].bar(['Gen','Stab','Gap'],vals,color=cols,alpha=0.7); axes[1,2].set_title('Diagnóstico'); axes[1,2].grid(alpha=0.3)
        
        # 8. Métricas Avanzadas Test
        adv=evaluation_results['test_performance']['advanced_metrics']
        names=['Median AE','Max Err','Abs Skew']
        vals=[adv['median_ae'],adv['max_error'],abs(adv['residual_skewness'])]
        axes[1,3].bar(names,vals,alpha=0.7); axes[1,3].set_title('Avanzadas Test'); axes[1,3].grid(alpha=0.3)
        
        # 9. Percentiles Error
        pct=['25%','50%','75%','90%','95%']
        tr=[evaluation_results['training_performance']['advanced_metrics']['mae_p25'],evaluation_results['training_performance']['advanced_metrics']['median_ae'],evaluation_results['training_performance']['advanced_metrics']['mae_p75'],evaluation_results['training_performance']['advanced_metrics']['mae_p90'],evaluation_results['training_performance']['advanced_metrics']['mae_p95']]
        te=[evaluation_results['test_performance']['advanced_metrics']['mae_p25'],evaluation_results['test_performance']['advanced_metrics']['median_ae'],evaluation_results['test_performance']['advanced_metrics']['mae_p75'],evaluation_results['test_performance']['advanced_metrics']['mae_p90'],evaluation_results['test_performance']['advanced_metrics']['mae_p95']]
        x=np.arange(len(pct)); w=0.35
        axes[2,0].bar(x-w/2,tr,w,label='Train'); axes[2,0].bar(x+w/2,te,w,label='Test'); axes[2,0].set_xticks(x); axes[2,0].set_xticklabels(pct); axes[2,0].set_title('Percentiles Err'); axes[2,0].legend(); axes[2,0].grid(alpha=0.3)
        
        # 10. Normality p-values
        nor=[]; labs=[]
        res_an=evaluation_results['test_performance']['residual_analysis']
        if 'normality_shapiro' in res_an: nor.append(res_an['normality_shapiro']['p_value']); labs.append('Shapiro')
        if 'normality_jarque_bera' in res_an: nor.append(res_an['normality_jarque_bera']['p_value']); labs.append('Jarque')
        if nor:
            cols=['green' if p>0.05 else 'red' for p in nor]
            axes[2,1].bar(labs,nor,color=cols,alpha=0.7); axes[2,1].axhline(0.05,linestyle='--'); axes[2,1].set_title('Normality Tests'); axes[2,1].grid(alpha=0.3)
        else:
            axes[2,1].text(0.5,0.5,'No tests',ha='center',va='center',transform=axes[2,1].transAxes)
        
        # 11. Homoscedasticity BP
        if 'homoscedasticity_bp' in res_an:
            bp=res_an['homoscedasticity_bp']
            color='green' if bp['is_homoscedastic'] else 'red'
            axes[2,2].bar(['BP'],[bp['p_value']],color=color,alpha=0.7); axes[2,2].axhline(0.05,linestyle='--'); axes[2,2].set_title('Homoscedasticity'); axes[2,2].grid(alpha=0.3)
        else:
            axes[2,2].text(0.5,0.5,'No BP',ha='center',va='center',transform=axes[2,2].transAxes)
        
        # 12. Placeholder for additional analysis
        axes[2,3].text(0.5,0.5,'Custom plots',ha='center',va='center',transform=axes[2,3].transAxes)
        axes[2,3].set_title('Additional')
        
        plt.tight_layout(rect=[0,0.03,1,0.95])
        if save_path:
            fig.savefig(save_path)
        else:
            plt.show()
