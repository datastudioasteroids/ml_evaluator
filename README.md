# ML Evaluator 🚀

## Descripción
**ML Evaluator** es un framework ligero y modular para **entrenar**, **evaluar** y **visualizar** pipelines de Machine Learning de forma reproducible. Permite comparar múltiples modelos, ensamblar resultados y exponerlos vía una interfaz (CLI o API), agilizando el ciclo de vida del ML en tus proyectos.

---

## ✨ Virtudes y Características Principales

- **Modularidad total**: separa claramente carga de datos, entrenamiento, evaluación y servicio.  
- **Ensamblado de modelos**: combina predicciones y métricas de distintos algoritmos para mejorar la robustez.  
- **Visualización automática**: genera gráficos de desempeño (curvas, distribuciones) sin configuraciones extra.  
- **CLI y API Ready**: ejecuta todo con un solo comando (`run.py`) o integra el `backend/` en tu aplicación.  
- **Reproducible**: logs detallados, ejemplos de CSV y plots incluidos.  
- **Fácil de extender**: añade nuevos modelos o métricas en segundos.

---

## 🔧 Cómo Funciona (a alto nivel)

1. **Preprocesado**: coloca tus datos en `data/` o conecta tu fuente personalizada.  
2. **Entrenamiento**: ejecuta `python run.py` para lanzar el pipeline completo.  
3. **Evaluación**: el sistema entrena todos los modelos definidos, guarda métricas en CSV y gráficos en `ensemble_Quantity_results_plots.png`.  
4. **Servicio**: importa `backend/` en tu API (Flask/FastAPI) para exponer endpoints de consulta.

---

## 🎯 Casos de Uso y Beneficios

- **Comparativa de Modelos**: prueba Random Forest, SVM y XGBoost con un solo script.  
- **Dashboards Automáticos**: genera reportes periódicos para equipos de Data Science.  
- **Educación y Formación**: ideal para clases de ML que muestran todo el ciclo de desarrollo.  
- **Microservicio de Predicciones**: levanta rápidamente una API REST con tu pipeline ya configurado.

---

## 💎 ¿Por Qué es Valioso este Proyecto?

- **Ahorra horas** de configuración y debugging al integrar las piezas clave de un pipeline ML.  
- **Reduce errores** humanos centralizando la lógica de entreno y evaluación.  
- **Agiliza el feedback** al equipo de negocio con reportes y gráficos listos para compartir.

**ML Evaluator** – tu aliado para llevar pipelines de ML de la idea a la producción con confianza y rapidez. 

---
