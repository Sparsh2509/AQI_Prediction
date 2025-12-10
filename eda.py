import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

plt.style.use("default")

# 1. Load Dataset
df = pd.read_csv("Dataset/station_hour.csv")
print("Shape:", df.shape)
print(df.head())

# 2. Basic Info
print(df.info())
print(df.describe(include="all"))

# 3. Missing values
print("Missing values:\n", df.isnull().sum())

# 4. Fill numeric missing
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 5. Remove duplicates
df = df.drop_duplicates()

# 6. Convert Datetime column
df['Datetime'] = pd.to_datetime(df['Datetime'], errors="coerce")

# Extract date features
df["year"] = df["Datetime"].dt.year
df["month"] = df["Datetime"].dt.month
df["day"] = df["Datetime"].dt.day
df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.day_name()

# 7. Pollutant Distributions
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]

for col in pollutants:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(12,7))
sns.heatmap(df[pollutants].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Pollutants")
plt.show()

# 9. AQI Trend Over Time
plt.figure(figsize=(14,5))
plt.plot(df["Datetime"], df["AQI"], linewidth=0.7)
plt.title("AQI Trend")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.grid(alpha=0.3)
plt.show()

# 10. Monthly AQI Trend
monthly_aqi = df.groupby(["year", "month"])["AQI"].mean()

plt.figure(figsize=(12,5))
monthly_aqi.plot()
plt.title("Average Monthly AQI")
plt.ylabel("AQI")
plt.show()

# 11. Station-wise Average AQI
station_aqi = df.groupby("Station")["AQI"].mean().sort_values()

plt.figure(figsize=(12,6))
station_aqi.plot(kind="bar", color="skyblue")
plt.title("Average AQI by Station")
plt.ylabel("AQI")
plt.show()

# 12. AQI Heatmap (Hour vs Weekday)
pivot = df.pivot_table(values="AQI", index="hour", columns="dayofweek", aggfunc="mean")

plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="viridis")
plt.title("AQI Variation by Hour and Day")
plt.show()

# 13. PM vs AQI Scatter Plots
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["PM2.5"], y=df["AQI"], alpha=0.5)
plt.title("PM2.5 vs AQI")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["PM10"], y=df["AQI"], alpha=0.5)
plt.title("PM10 vs AQI")
plt.show()


# 5. BOXPLOTS FOR OUTLIERS

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "AQI"]
plt.figure(figsize=(10,5))
sns.boxplot(data=df[pollutants])
plt.title("Outliers in Pollutants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# STATION-WISE MONTHLY HEATMAP

pivot = df.pivot_table(
    values="AQI",
    index="Station",
    columns=df["Datetime"].dt.month,
    aggfunc="mean"
)

plt.figure(figsize=(12,5))
sns.heatmap(pivot, cmap="coolwarm", annot=True, fmt=".1f")
plt.title("Monthly AQI by Station")
plt.tight_layout()
plt.show()


# RANDOM FOREST FEATURE IMPORTANCE

features = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO",
            "SO2","O3","Benzene","Toluene","Xylene"]
available_features = [f for f in features if f in df.columns]

X = df[available_features]
y = df["AQI"]

model = RandomForestRegressor(random_state=42).fit(X, y)
importance = pd.Series(model.feature_importances_, index=available_features)

plt.figure(figsize=(6,5))
importance.sort_values().plot(kind='barh')
plt.title("Pollutant Feature Importance")
plt.tight_layout()
plt.show()

print("FULL EDA Completed Successfully!")