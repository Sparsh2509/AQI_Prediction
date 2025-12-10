# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")

# 2. Load Dataset
df = pd.read_csv("Dataset/station_hour.csv")
print("Shape:", df.shape)
print(df.head())

# 3. Basic Info
print(df.info())
print(df.describe(include="all"))

# 4. Check Missing Values
print("Missing values:\n", df.isnull().sum())

# Fill numeric missing with median
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Drop rows with excessive missing values
df = df.dropna(thresh=int(0.5 * len(df.columns)))
print("After cleaning shape:", df.shape)

# 5. Remove duplicates
df = df.drop_duplicates()

# 6. Convert date column
df['date'] = pd.to_datetime(df['date'], errors="coerce")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["hour"] = df["date"].dt.hour
df["dayofweek"] = df["date"].dt.day_name()

# 7. Pollutant Distributions
pollutants = ["pm2_5", "pm10", "no2", "so2", "co", "o3", "aqi"]

for col in pollutants:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[pollutants].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Pollutants")
plt.show()

# 9. Daily AQI Trend
daily_aqi = df.groupby("date")["aqi"].mean()

plt.figure(figsize=(14,5))
plt.plot(daily_aqi)
plt.title("Daily AQI Trend")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.show()

# 10. Monthly AQI Trend
monthly_aqi = df.groupby(["year", "month"])["aqi"].mean()

plt.figure(figsize=(12,5))
monthly_aqi.plot()
plt.title("Monthly AQI Trend (2015–2024)")
plt.ylabel("AQI")
plt.show()

# 11. Station-wise Average AQI
station_aqi = df.groupby("station")["aqi"].mean().sort_values(ascending=False).head(20)

plt.figure(figsize=(12,6))
station_aqi.plot(kind="bar", color="skyblue")
plt.title("Top 20 Most Polluted Stations")
plt.ylabel("Average AQI")
plt.show()

# 12. AQI Heatmap (Hour vs Day of Week)
pivot = df.pivot_table(values="aqi", index="hour", columns="dayofweek", aggfunc="mean")

plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="viridis")
plt.title("AQI Variation by Hour and Day")
plt.show()

# 13. PM vs AQI Scatter Plots
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["pm2_5"], y=df["aqi"], alpha=0.5)
plt.title("PM2.5 vs AQI")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["pm10"], y=df["aqi"], alpha=0.5)
plt.title("PM10 vs AQI")
plt.show()

print("✔ FULL EDA Completed Successfully!")