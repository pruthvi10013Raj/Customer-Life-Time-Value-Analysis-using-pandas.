import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("E:\\Projects\\CLV_python\\data")
print("Current Directory:", os.getcwd())

#loding data
df = pd.read_excel("data.xlsx")
print("Data loaded successfully!")
print(df.head())

#cleaning using dropna, remove rows where the CustomerID is missing.
#count reduces from 541909 to 406829
df.dropna(subset=["CustomerID"], inplace=True)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

#We are converting the transaction data into monthly summaries by grouping all invoices by month-end dates
# and calculating the total sales amount for each month. This transforms daily transaction data
# into a time series of monthly sales totals for trend analysis, which you will see in the plotted graphs further.
monthly_sales = df.set_index("InvoiceDate").resample("ME")["TotalSales"].sum()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
plt.title("Monthly Sales Over Time")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

