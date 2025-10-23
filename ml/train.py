import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load CSV (update path)
path = "D:/projects/neurosync/ml/ml_ready_data.csv"
df = pd.read_csv(path)

# Drop any rows with missing values
df = df.dropna()

# Features & target
X = df[['focus_percent', 'blink_per_min', 'gaze_x', 'gaze_y', 'yaw', 'pitch', 'drift_count']]
y = df['cognitive_state']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model & label encoder
joblib.dump(clf, "ml/model.pkl")
joblib.dump(le, "ml/label_encoder.pkl")
print("✅ Model and label encoder saved in ml/")
