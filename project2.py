import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================
# Load the existing model and scaler
# ============================
def load_model():
    model = joblib.load('three_phase_fault_model.pkl')
    scaler = joblib.load('three_phase_fault_scaler.pkl')
    return model, scaler

# ============================
# Save the updated model and scaler
# ============================
def save_model(model, scaler):
    joblib.dump(model, 'three_phase_fault_model.pkl')
    joblib.dump(scaler, 'three_phase_fault_scaler.pkl')
    print("Model and scaler updated and saved!")

# ============================
# Dynamically update the model with new data
# ============================
def update_model_with_new_data(new_data, model, scaler):
    # Preprocess new data
    X_new = new_data[['Va', 'Vb', 'Vc']]
    y_new = new_data['Faulty']

    # Feature scaling using the existing scaler
    X_new_scaled = scaler.transform(X_new)

    # Update the model with new data
    model.fit(X_new_scaled, y_new)

    # Save the updated model and scaler
    save_model(model, scaler)

# ============================
# Step 1: Load Dataset from Excel
# ============================
def load_dataset(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Check if required columns are present
    if not all(col in df.columns for col in ['Va', 'Vb', 'Vc', 'Faulty']):
        raise ValueError("Excel file must contain columns: Va, Vb, Vc, Faulty")

    return df

# ============================
# Step 2: Data Preprocessing
# ============================
def preprocess_data(df):
    X = df[['Va', 'Vb', 'Vc']]  # Features
    y = df['Faulty']            # Target

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

# ============================
# Step 3: Model Building
# ============================
def build_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

# ============================
# Step 4: Model Evaluation
# ============================
def evaluate_model(model, X_test, y_test):
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
def save_trained_model(model, scaler):
    joblib.dump(model, 'three_phase_fault_model.pkl')
    joblib.dump(scaler, 'three_phase_fault_scaler.pkl')
    print("\nModel and Scaler saved successfully!")

# ============================
# Step 6: Prediction Function
# ============================
def predict_fault(Va, Vb, Vc):
    model, scaler = load_model()  # Load the latest model and scaler
    input_data = np.array([[Va, Vb, Vc]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)

    if pred[0] == 1:
        return "Faulty"
    else:
        return "No Fault"

# ============================
# Example of using the code with new data and predictions
# ============================
# Load the initial dataset from an Excel file
file_path = '/Users/sayeedanwar/Desktop/project/three_phase_fault_data_with_serial_first.xlsx'  # Replace with your file path
df = load_dataset(file_path)

# Preprocess the data
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Build and evaluate the initial model
model = build_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

# Save the trained model and scaler
save_trained_model(model, scaler)

# Example: Dynamically update the model with new data
new_data = pd.DataFrame({
    'Va': [380, 430, 415],
    'Vb': [370, 440, 420],
    'Vc': [390, 450, 430],
    'Faulty': [1, 0, 0]
})

# Load the existing model and scaler
model, scaler = load_model()

# Update the model with new data
update_model_with_new_data(new_data, model, scaler)

# Example prediction after updating the model
new_Va = 380
new_Vb = 370
new_Vc = 390
predicted_fault = predict_fault(new_Va, new_Vb, new_Vc)
print(f"\nPredicted Condition for given inputs: {predicted_fault}")
