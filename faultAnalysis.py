# three_phase_fault_detection.py

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================
# Step 1: Load Dataset
# ============================
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    required_cols = ['Va', 'Vb', 'Vc', 'Status']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Excel file must contain columns: {required_cols}")
    return df

# ============================
# Step 2: Preprocess Data
# ============================
def preprocess_data(df):
    X = df[['Va', 'Vb', 'Vc']]
    y = df['Status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM input: (samples, timesteps, features)
    X_scaled = np.expand_dims(X_scaled, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# ============================
# Step 3: Build BiLSTM Model
# ============================
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False), input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================
# Step 4: Train and Evaluate Model
# ============================
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc*100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model

# ============================
# Step 5: Save Model and Scaler
# ============================
def save_model(model, scaler):
    model.save('three_phase_fault_model.keras')
    joblib.dump(scaler, 'three_phase_fault_scaler.pkl')
    print("\nModel and Scaler saved successfully!")

# ============================
# Step 6: Load Model and Scaler
# ============================
def load_model_and_scaler():
    model = tf.keras.models.load_model('three_phase_fault_model.keras')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Re-compile for training
    scaler = joblib.load('three_phase_fault_scaler.pkl')
    return model, scaler

# ============================
# Step 7: Update Model with New Data
# ============================
def update_model_with_new_data(new_data, model, scaler):
    X_new = new_data[['Va', 'Vb', 'Vc']]
    y_new = new_data['Faulty']

    X_new_scaled = scaler.transform(X_new)
    X_new_scaled = np.expand_dims(X_new_scaled, axis=1)

    model.fit(X_new_scaled, y_new, epochs=5, batch_size=1, verbose=1)

    save_model(model, scaler)

# ============================
# Step 8: Predict Fault
# ============================
def predict_fault(Va, Vb, Vc):
    model, scaler = load_model_and_scaler()

    input_data = np.array([[Va, Vb, Vc]])
    input_scaled = scaler.transform(input_data)
    input_scaled = np.expand_dims(input_scaled, axis=1)

    pred_prob = model.predict(input_scaled)
    pred = (pred_prob > 0.5).astype(int)

    return "Faulty" if pred[0][0] == 1 else "No Fault"

# ============================
# Main Execution Example
# ============================
if __name__ == "__main__":
    # 1. Load and preprocess data
    file_path = '/Users/sayeedanwar/Desktop/project/Book2.xlsx'
    df = load_dataset(file_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # 2. Train and evaluate model
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # 3. Save model and scaler
    save_model(model, scaler)

    # 4. Update model with new data
    new_data = pd.DataFrame({
        'Va': [380, 430, 415],
        'Vb': [370, 440, 420],
        'Vc': [390, 450, 430],
        'Faulty': [1, 0, 0]
    })

    model, scaler = load_model_and_scaler()
    update_model_with_new_data(new_data, model, scaler)

    # 5. Make a prediction
    new_Va = 16.99
    new_Vb = 98.56
    new_Vc = 101.42
    result = predict_fault(new_Va, new_Vb, new_Vc)
    print(f"\nPredicted Condition: {result}")
