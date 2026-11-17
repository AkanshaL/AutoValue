
import re
import os
import joblib
import optuna
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD DATASET

DATA_PATH = "dataset/used_cars.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# 2. CLEAN NUMERICAL COLUMNS

def extract_number(value):
    if pd.isna(value):
        return np.nan

    match = re.search(r"\d+\.?\d*", str(value))

    if match:
        return float(match.group())

    return np.nan


numeric_columns = [
    "mileage",
    "engine",
    "max_power",
    "seats"
]

for col in numeric_columns:
    df[col] = df[col].apply(extract_number)


# 3. SELECT FEATURES

features = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

target = "selling_price"

X = df[features].copy()
y = df[target].copy()

# Remove invalid target values
valid = y.notna() & (y > 0)

X = X.loc[valid]
y = y.loc[valid]


# 4. DEFINE COLUMN TYPES

categorical_features = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

numerical_features = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]


# 5. PREPROCESSING

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# 6. TRAIN AND TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# 7. OPTUNA HYPERPARAMETER OPTIMIZATION

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 600
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),

        "num_leaves": trial.suggest_int(
            "num_leaves", 15, 63
        ),

        "min_child_samples": trial.suggest_int(
            "min_child_samples", 10, 40
        ),

        "subsample": trial.suggest_float(
            "subsample", 0.7, 1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.1, 10.0, log=True
        ),

        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1
    }

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LGBMRegressor(**params))
    ])

    X_train_part, X_valid, y_train_part, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42
    )

    model_pipeline.fit(X_train_part, y_train_part)

    predictions = model_pipeline.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)

    return mae


print("\nStarting Optuna optimization...")

study = optuna.create_study(direction="minimize")

study.optimize(objective, n_trials=20)

print("\nBest Optuna parameters:")
print(study.best_params)


# 8. TRAIN FINAL MODEL

best_model = LGBMRegressor(
    **study.best_params,
    verbosity=-1,
    random_state=42,
    n_jobs=-1
)

final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", best_model)
])

final_pipeline.fit(X_train, y_train)


# 9. MODEL EVALUATION

predictions = final_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))
print("===================================")


# 10. SAVE TRAINED MODEL

os.makedirs("models", exist_ok=True)

joblib.dump(final_pipeline, "models/car_price_model.pkl")

joblib.dump(features, "models/features.pkl")

print("\nModel saved successfully!")
print("Location: models/car_price_model.pkl")
