# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Define the dataset manually
data_dict = {
    'PM2.5': [55, 78, 90, 50, 85, 60, 75, 82, 68, 74],
    'PM10': [120, 140, 135, 100, 130, 115, 125, 132, 118, 128],
    'NO2': [45, 55, 60, 40, 58, 48, 52, 54, 50, 53],
    'temperature': [30, 32, 29, 31, 33, 30, 31, 32, 30, 31],
    'humidity': [65, 70, 68, 64, 69, 66, 67, 71, 65, 68],
    'AQI': [150, 175, 190, 140, 180, 160, 170, 182, 158, 165]
}

# Convert the dictionary to a DataFrame
data = pd.DataFrame(data_dict)

# Step 2: Data Preprocessing (already clean, no missing values here)

# Step 3: Exploratory Data Analysis (Optional)
# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 4: Feature Selection
# Define features (X) and target variable (y)
X = data[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
y = data['AQI']

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

# Step 9: Visualize the results
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()


importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()