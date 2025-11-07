model trainig___

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_models(rfm_train):
    """Train Linear Regression and Random Forest models."""
    X = rfm_train[["Recency", "Frequency", "Monetary"]]
    y = rfm_train["FutureMonetary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linreg = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    y_pred_lr = linreg.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    print(f"\nLinear Regression RMSE: {rmse_lr:.2f}")
    print(f"Random Forest RMSE: {rmse_rf:.2f}")

    return (linreg, rf, X_test, y_test, y_pred_lr, y_pred_rf)