import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 1. Load Dataset
df = pd.read_csv(r"D:\Sparsh\ML_Projects\AQI_Prediction\Dataset\station_hour.csv")


# 5. Select features and target
features = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO",
            "SO2","O3","Benzene","Toluene","Xylene"]

available_features = [f for f in features if f in df.columns]

X = df[available_features]
y = df["AQI"]

# 6. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Random Forest R²: {r2:.3f}")
print(f"Random Forest RMSE: {rmse:.3f}")

# 9. Save trained model
joblib.dump(rf_model, "aqi_rf_model.pkl")
print("Random Forest model saved as 'aqi_rf_model.pkl'")
