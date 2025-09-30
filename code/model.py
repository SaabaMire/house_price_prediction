from utils import prepare_features_from_raw
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load  cleaned dataset
CSV_PATH = "dataset/clean_house_dataset.csv"
df = pd.read_csv(CSV_PATH)


X = df.drop(columns=["Price", "LogPrice"])
y = df["Price"]

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Linear Regression
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)

#  Random Forest
rf = RandomForestRegressor(
    n_estimators=100, random_state=42).fit(X_train, y_train)
rf_pred = rf.predict(X_test)

#  Metrics


def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n{name} Performance:")
    print(f"  R²   : {r2:.3f}")
    print(f"  MAE  : {mae:,.0f}")
    print(f"  MSE  : {mse:,.0f}")
    print(f"  RMSE : {rmse:,.0f}")


print_metrics("Linear Regression", y_test, lr_pred)
print_metrics("Random Forest",   y_test, rf_pred)

# Single-row sanity check
i = 7
x_one_df = X_test.iloc[[i]]
y_true = y_test.iloc[i]
p_lr_one = float(lr.predict(x_one_df)[0])
p_rf_one = float(rf.predict(x_one_df)[0])
print("\nSingle-row sanity check:")
print(f"  Actual Price: ${y_true:,.0f}")
print(f"  LR Pred     : ${p_lr_one:,.0f}")
print(f"  RF Pred     : ${p_rf_one:,.0f}")

# Saving models
joblib.dump(lr, "models/lr_model.joblib")
joblib.dump(rf, "models/rf_model.joblib")
print("\nSaved models → models/lr_model.joblib and models/rf_model.joblib")

# Example of using the prepare_features_from_raw() function
custom = {
    "Size_sqft": 2400, "Bedrooms": 3, "Bathrooms": 2,
    "YearBuilt": 2010, "Location": "City"
}
x_new_df = prepare_features_from_raw(custom)
print("\n=== Custom Input Prediction ===")
print("Linear Regression:", float(lr.predict(x_new_df)[0]))
print("Random Forest    :", float(rf.predict(x_new_df)[0]))
