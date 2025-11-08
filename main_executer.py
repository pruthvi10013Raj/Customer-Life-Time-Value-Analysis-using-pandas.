from src.data_preprocessing import load_and_clean_data, calculate_rfm, create_train_test_sets
from src.model_training import train_models
from src.visualization import (
    plot_monthly_sales, plot_predictions,
    plot_top_products, plot_top_countries
)

# --- LOAD & CLEAN ---
df = load_and_clean_data("E:\\Projects\\CLV_python\\data\\data.xlsx")
plot_monthly_sales(df)

# --- RFM ---
rfm, snapshot_date = calculate_rfm(df)
rfm_train = create_train_test_sets(df, rfm)

# --- MODEL TRAINING ---
linreg, rf, X_test, y_test, y_pred_lr, y_pred_rf = train_models(rfm_train)

# --- VISUALIZATION ---
plot_predictions(y_test, y_pred_lr, "Linear Regression", "blue")
plot_predictions(y_test, y_pred_rf, "Random Forest", "green")
plot_top_products(df)
plot_top_countries(df)

print("\n✅ CLV Analysis Completed Successfully!")
