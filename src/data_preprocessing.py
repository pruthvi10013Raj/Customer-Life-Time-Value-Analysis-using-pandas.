data preprocessing___

import pandas as pd
import os

def load_and_clean_data(data_path):
    """Load Excel data and perform initial cleaning."""
    os.chdir(os.path.dirname(data_path))
    df = pd.read_excel(data_path)
    df.dropna(subset=["CustomerID"], inplace=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalSales"] = df["Quantity"] * df["UnitPrice"]
    return df

def calculate_rfm(df):
    """Calculate Recency, Frequency, and Monetary metrics."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalSales": "sum"
    })
    rfm.rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalSales": "Monetary"
    }, inplace=True)
    return rfm, snapshot_date

def create_train_test_sets(df, rfm):
    """Split customers into training and test data by date."""
    cutoff_date = df["InvoiceDate"].max() - pd.DateOffset(months=2)
    train = df[df["InvoiceDate"] <= cutoff_date]
    test = df[df["InvoiceDate"] > cutoff_date]
    train_monetary = train.groupby("CustomerID")["TotalSales"].sum()
    test_monetary = test.groupby("CustomerID")["TotalSales"].sum()

    rfm_train = rfm.copy()
    rfm_train["FutureMonetary"] = test_monetary
    rfm_train["FutureMonetary"].fillna(0, inplace=True)
    return rfm_train