import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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

# 4. Fill numeric missing values with median
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 5. Remove duplicates
df = df.drop_duplicates()

# 6. Convert Datetime column and extract features
df['Datetime'] = pd.to_datetime(df['Datetime'], errors="coerce")
df["year"] = df["Datetime"].dt.year
df["month"] = df["Datetime"].dt.month
df["day"] = df["Datetime"].dt.day
df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.day_name()

# 7. Key pollutants for analysis
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]

# Keep only available columns
available_features = [f for f in pollutants if f in df.columns and f != "AQI"]

# 8. Distribution plots
for col in ["PM2.5", "PM10", "NO2", "SO2", "AQI"]:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.show()

# 9. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df[available_features + ["AQI"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Pollutants and AQI")
plt.show()

# 10. AQI Trend Over Time
plt.figure(figsize=(12,5))
plt.plot(df["Datetime"], df["AQI"], linewidth=0.7)
plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.grid(alpha=0.3)
plt.show()

# 11. Station-wise Average AQI
station_aqi = df.groupby("Station")["AQI"].mean().sort_values()
plt.figure(figsize=(10,5))
station_aqi.plot(kind="bar", color="skyblue")
plt.title("Average AQI by Station")
plt.ylabel("AQI")
plt.show()

# 12. Boxplots for Outliers
plt.figure(figsize=(8,5))
sns.boxplot(data=df[["PM2.5", "PM10", "NO2", "SO2", "AQI"]])
plt.title("Outliers in Key Pollutants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 13. AQI Heatmap (Hour vs Weekday)
pivot = df.pivot_table(values="AQI", index="hour", columns="dayofweek", aggfunc="mean")
plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="viridis")
plt.title("AQI Variation by Hour and Day")
plt.show()

# 14. Random Forest & XGBoost
X = df[available_features]
y = df["AQI"]

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X, y)
rf_importance = pd.Series(rf_model.feature_importances_, index=available_features).sort_values()

plt.figure(figsize=(6,5))
rf_importance.plot(kind='barh', color="skyblue")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# XGBoost
xgb_model = XGBRegressor(n_estimators=200, random_state=42, learning_rate=0.1)
xgb_model.fit(X, y)
xgb_importance = pd.Series(xgb_model.feature_importances_, index=available_features).sort_values()

plt.figure(figsize=(6,5))
xgb_importance.plot(kind='barh', color="lightgreen")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

print("EDA + Feature Importance with Random Forest & XGBoost Completed!")
