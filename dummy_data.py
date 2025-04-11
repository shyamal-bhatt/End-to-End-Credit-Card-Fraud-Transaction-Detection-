import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate 100 business days starting from 2023-01-01
dates: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=100, freq="B")

# Simulate stock prices using a random walk
start_price: float = 100.0
prices: list[float] = [start_price]

# Generate daily percentage changes with small fluctuations
for _ in range(1, len(dates)):
    # Daily change drawn from a normal distribution with mean 0 and std 0.02
    change: float = np.random.normal(0, 0.02)
    new_price: float = prices[-1] * (1 + change)
    prices.append(new_price)

# Create a DataFrame with the generated data
df: pd.DataFrame = pd.DataFrame({"Date": dates, "Close": prices})

# Save the dummy dataset to a CSV file
csv_filename: str = "dummy_stock_data.csv"
df.to_csv(csv_filename, index=False)

print(f"Dummy stock market dataset created and saved as {csv_filename}!")
print(df.head())
