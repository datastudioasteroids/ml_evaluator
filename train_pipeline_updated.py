# train_pipeline_updated.py

import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from ml_utils import normalize_columns, extract_date_features  # <— IMPORTAMOS AQUÍ

def build_preprocessor() -> ColumnTransformer:
    date_pipe = Pipeline([
        ("extract", FunctionTransformer(extract_date_features, validate=False)),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    region_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    product_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([
        ("date",    date_pipe,    ["date"]),
        ("region",  region_pipe,  ["region"]),
        ("product", product_pipe, ["product"]),
    ], remainder="drop")

def build_pipeline(model_params: dict):
    return Pipeline([
        ("preproc", build_preprocessor()),
        ("scale",   StandardScaler(with_mean=False)),
        ("model",   xgb.XGBRegressor(**model_params))
    ])

def train_and_save(data_path: str, out_dir: str, model_params: dict = None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Leer y parsear fecha
    df = pd.read_csv(data_path, parse_dates=["Order Date"], encoding="latin1")
    df = df.rename(columns={"Order Date": "date"}, errors="ignore")

    # Normalizar nombres y renombrar targets
    df = normalize_columns(df)
    df = df.rename(columns={
        "Region": "region",
        "Product Name": "product",
        "Quantity": "quantity",
        "Profit": "profit"
    }, errors="ignore")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("La columna 'date' no parseó correctamente.")

    X = df[["date", "region", "product"]]
    y_p = df["profit"]
    y_q = df["quantity"]

    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "random_state": 42,
        "n_jobs": -1
    }
    params = model_params or default_params

    pipe_p = build_pipeline(params)
    pipe_q = build_pipeline(params)
    pipe_p.fit(X, y_p)
    pipe_q.fit(X, y_q)

    joblib.dump(pipe_p, out_path / "pipeline_profit.pkl")
    joblib.dump(pipe_q, out_path / "pipeline_quantity.pkl")
    print(f"✅ Artefactos guardados en {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=r"D:\Repositorios\Modelos_ML\data\stores_sales_forecasting.csv"
    )
    parser.add_argument(
        "--outdir",
        default=r"D:\Repositorios\Modelos_ML\backend\models_features"
    )
    args = parser.parse_args()
    train_and_save(args.data, args.outdir)
