# --------------------------
# IMPORT LIBRARIES
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor

# --------------------------
# LOAD DATA
# --------------------------
df = pd.read_csv("Dataset/station_hour.csv")

# Convert Datetime column
df["Datetime"] = pd.to_datetime(df["Datetime"])

print("Shape:", df.shape)
print(df.head())
print(df.info())

# --------------------------
# BASIC STATS
# --------------------------
print("\nMissing Values:\n", df.isna().sum())
print("\nDescribe:\n", df.describe(include="all"))


# ----------------------------------------------------------
# 1. AQI BUCKET DISTRIBUTION
# ----------------------------------------------------------
plt.figure(figsize=(8,4))
sns.countplot(data=df, x="AQI_Bucket", palette="viridis")
plt.title("AQI Bucket Distribution")
plt.xticks(rotation=45)
plt.show()


# ----------------------------------------------------------
# 2. CITY-WISE AQI COMPARISON
# ----------------------------------------------------------
plt.figure(figsize=(8,4))
df.groupby("City")["AQI"].mean().sort_values().plot(kind="bar")
plt.title("Average AQI by City")
plt.ylabel("AQI")
plt.show()


# ----------------------------------------------------------
# 3. TIME SERIES DECOMPOSITION (DAILY AQI TREND)
# ----------------------------------------------------------
daily = df.set_index("Datetime")["AQI"].resample("D").mean()

result = seasonal_decompose(daily, model="additive", period=30)
result.plot()
plt.show()


# ----------------------------------------------------------
# 4. POLLUTANT TREND OVER TIME
# ----------------------------------------------------------
plt.figure(figsize=(12,6))
for col in ["PM2.5","PM10","SO2","NO2","O3"]:
    plt.plot(df["Datetime"], df[col], label=col, alpha=0.6)
plt.legend()
plt.title("Major Pollutants Trend Over Time")
plt.show()


# ----------------------------------------------------------
# 5. BOXPLOTS FOR OUTLIERS
# ----------------------------------------------------------
plt.figure(figsize=(10,5))
sns.boxplot(data=df[["PM2.5", "PM10", "NO2", "SO2", "AQI"]])
plt.title("Outliers in Pollutants")
plt.xticks(rotation=45)
plt.show()


# ----------------------------------------------------------
# 6. SCATTERPLOT – AQI VS POLLUTANTS
# ----------------------------------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["PM2.5"], y=df["AQI"], alpha=0.4)
plt.title("PM2.5 vs AQI")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["PM10"], y=df["AQI"], alpha=0.4)
plt.title("PM10 vs AQI")
plt.show()


# ----------------------------------------------------------
# 7. POLLUTANT CONTRIBUTION PIE CHART
# ----------------------------------------------------------
plt.figure(figsize=(6,6))
df[["PM2.5","PM10","NO2","SO2","CO"]].mean().plot(
    kind="pie", autopct='%1.1f%%')
plt.title("Average Pollutant Contribution")
plt.ylabel("")
plt.show()


# ----------------------------------------------------------
# 8. AQI HOURLY & WEEKLY PATTERN
# ----------------------------------------------------------
df["hour"] = df["Datetime"].dt.hour
df["weekday"] = df["Datetime"].dt.day_name()

plt.figure(figsize=(8,4))
df.groupby("hour")["AQI"].mean().plot()
plt.title("Hourly AQI Pattern")
plt.show()

plt.figure(figsize=(8,4))
df.groupby("weekday")["AQI"].mean().plot(kind="bar")
plt.title("AQI by Day of Week")
plt.xticks(rotation=45)
plt.show()


# ----------------------------------------------------------
# 9. STATION-WISE MONTHLY HEATMAP
# ----------------------------------------------------------
pivot = df.pivot_table(
    values="AQI",
    index="Station",
    columns=df["Datetime"].dt.month,
    aggfunc="mean"
)

plt.figure(figsize=(12,5))
sns.heatmap(pivot, cmap="coolwarm")
plt.title("Monthly AQI by Station")
plt.show()


# ----------------------------------------------------------
# 10. RANDOM FOREST FEATURE IMPORTANCE
# ----------------------------------------------------------
features = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO",
            "SO2","O3","Benzene","Toluene","Xylene"]

X = df[features]
y = df["AQI"]

model = RandomForestRegressor().fit(X, y)

importance = pd.Series(model.feature_importances_, index=features)

plt.figure(figsize=(6,5))
importance.sort_values().plot(kind='barh')
plt.title("Pollutant Feature Importance")
plt.show()


# --------------------------
# END OF FULL EDA
# --------------------------
