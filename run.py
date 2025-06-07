from ml_pipeline.main_optimized_training import OptimizedSalesPredictor

if __name__ == "__main__":
    # Directorio base donde está el CSV y la carpeta ml_pipeline
    BASE_DIR = r"D:\Repositorios\Modelos_ML"
    
    # Puedes ajustar la configuración si lo deseas
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
    
    predictor = OptimizedSalesPredictor(base_dir=BASE_DIR, config=config)
    df = predictor.load_and_prepare_data()
    
    # Entrenar para Quantity
    model_q, metrics_q = predictor.train_optimized_model(target_col="Quantity")
    # Entrenar para Profit
    model_p, metrics_p = predictor.train_optimized_model(target_col="Profit")
    
    # Guardar modelos y reporte
    predictor.save_models()
    report = predictor.generate_report()
    
    print("\n¡Proceso completado!")

