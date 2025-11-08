import matplotlib.pyplot as plt

def plot_monthly_sales(df):
    monthly_sales = df.set_index("InvoiceDate").resample("ME")["TotalSales"].sum()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
    plt.title("Monthly Sales Over Time")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.grid(True)
    plt.show()

def plot_predictions(y_test, y_pred, model_name, color):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color=color, label="Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", lw=2, label="Perfect Prediction")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.xlabel("Actual Future Monetary Value")
    plt.ylabel("Predicted Future Monetary Value")
    plt.legend()
    plt.show()

def plot_top_products(df):
    top_products = df.groupby("Description")["TotalSales"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_products.plot(kind="bar")
    plt.title("Top 10 Products by Sales")
    plt.xlabel("Product")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def plot_top_countries(df):
    top_countries = df.groupby("Country")["TotalSales"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_countries.plot(kind="bar", color="skyblue")
    plt.title("Top 10 Countries by Sales")
    plt.xlabel("Country")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.show()
