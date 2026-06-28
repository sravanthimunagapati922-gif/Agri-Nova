import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
# Replace 'Crop_recommendation.csv' with your exact file path
data = pd.read_csv('Crop_recommendation.csv')

# 2. Separate features (X) and target (y)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the features for distance-based algorithms
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

rf_model.fit(X_train_scaled, y_train)
dt_model.fit(X_train_scaled, y_train)

# 6. Make predictions
rf_predictions = rf_model.predict(X_test_scaled)
dt_predictions = dt_model.predict(X_test_scaled)

# 7. Evaluate the models
print("--- Random Forest Evaluation ---")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("\nClassification Report:\n", classification_report(y_test, rf_predictions))

print("\n--- Decision Tree Evaluation ---")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("\nClassification Report:\n", classification_report(y_test, dt_predictions))

# 8. Function to predict the best crop based on new input
def recommend_crop(N, P, K, temp, humidity, ph, rainfall):
    input_data = scaler.transform([[N, P, K, temp, humidity, ph, rainfall]])
    prediction = rf_model.predict(input_data)
    return prediction[0]

# Example prediction: (N, P, K, Temperature, Humidity, pH, Rainfall)
# Note: Ensure parameter units and ranges match your CSV requirements
new_crop = recommend_crop(90, 42, 43, 20.87, 82.00, 6.5, 202.9)
print(f"\nThe recommended crop is: {new_crop}")
