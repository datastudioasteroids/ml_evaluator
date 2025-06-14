# predict_cli.py

import sys
import joblib
import pandas as pd
from datetime import datetime
from ml_utils import normalize_columns  # <— importamos también aquí

PIPE_QTY  = r"D:\Repositorios\Modelos_ML\backend\models_features\pipeline_quantity.pkl"
PIPE_PROF = r"D:\Repositorios\Modelos_ML\backend\models_features\pipeline_profit.pkl"

def prompt_input(prompt):
    val = input(prompt).strip()
    if not val:
        raise SystemExit("❌ Se requiere un valor.")
    return val

def main():
    try:
        pipe_q = joblib.load(PIPE_QTY)
        pipe_p = joblib.load(PIPE_PROF)
    except Exception as e:
        raise SystemExit(f"❌ Error cargando pipelines: {e}")

    prod   = prompt_input("Nombre de producto: ")
    region = prompt_input("Región: ")
    date_s = prompt_input("Fecha (YYYY-MM-DD): ")
    try:
        dt = datetime.strptime(date_s, "%Y-%m-%d")
    except ValueError:
        raise SystemExit("❌ Formato de fecha inválido.")

    df = pd.DataFrame([{"date": dt, "region": region, "product": prod}])
    df = normalize_columns(df)

    qty  = pipe_q.predict(df)[0]
    prof = pipe_p.predict(df)[0]

    print(f"\n➡️ Predicción para “{prod}” en {region} / {date_s[:7]}:")
    print(f"   • Cantidad: {qty:.2f}")
    print(f"   • Ganancia: ${prof:.2f}\n")

if __name__ == "__main__":
    main()
