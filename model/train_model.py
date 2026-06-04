import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset/crop_data.csv")

# Features and label
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
with open("model/crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")