# Power System Fault Analysis (Three Phase Voltage - Faulty/No Faulty)

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================
# Step 1: Load Dataset from Excel
# ============================

# Load data from an Excel file
def load_dataset(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Check if required columns are present
    if not all(col in df.columns for col in ['Va', 'Vb', 'Vc', 'Faulty']):
        raise ValueError("Excel file must contain columns: Va, Vb, Vc, Faulty")

    return df

# Load dataset from the given Excel file
file_path = '/Users/sayeedanwar/Desktop/project/three_phase_fault_data_with_serial_first.xlsx'  # Replace with your file path
df = load_dataset(file_path)

# ============================
# Step 2: Data Preprocessing
# ============================

X = df[['Va', 'Vb', 'Vc']]  # Features
y = df['Faulty']            # Target

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ============================
# Step 3: Model Building
# ============================

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ============================
# Step 4: Evaluation
# ============================

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")

# Confusion Matrix and Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================
# Step 5: Save Model and Scaler
# ============================

# Save model and scaler
joblib.dump(model, 'three_phase_fault_model.pkl')
joblib.dump(scaler, 'three_phase_fault_scaler.pkl')

print("\nModel and Scaler saved successfully!")

# ============================
# Step 6: Prediction Function
# ============================

def predict_fault(Va, Vb, Vc):
    # Load trained model and scaler
    model = joblib.load('three_phase_fault_model.pkl')
    scaler = joblib.load('three_phase_fault_scaler.pkl')

    input_data = np.array([[Va, Vb, Vc]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)

    if pred[0] == 1:
        return "Faulty"
    else:
        return "No Fault"

# ============================
# Step 7: Example Prediction
# ============================

# Example: Predict fault type for new input
new_Va = 380
new_Vb = 370
new_Vc = 360

predicted_fault = predict_fault(new_Va, new_Vb, new_Vc)
print(f"\nPredicted Condition for given inputs: {predicted_fault}")
