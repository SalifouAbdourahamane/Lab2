# app.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from flask import Flask, jsonify
import sys

app = Flask(__name__)

# Global storage
dataset = None
model = None
test_results = None

def load_dataset(filepath='iris_extended_encoded.csv'):
    print(f"Loading dataset from {filepath}", file=sys.stdout, flush=True)
    df = pd.read_csv(filepath, header=0)
    features = df.iloc[:, 1:21].values  # 20 features
    labels = df.iloc[:, 0].values       # First column as labels
    
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    dataset = {
        'features': features,
        'labels': encoded_labels,
        'label_encoder': le
    }
    print(f"Dataset loaded", file=sys.stdout, flush=True)
    return dataset

def build_model():
    print("Building model", file=sys.stdout, flush=True)
    model = Sequential([
        Dense(64, activation='relu', input_dim=20),
        Dense(3, activation='softmax')  # 3 classes for Iris
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(f"Model built", file=sys.stdout, flush=True)
    return model

def train_model(model, dataset):
    print("Training model", file=sys.stdout, flush=True)
    model.fit(dataset['features'], dataset['labels'], epochs=10, verbose=0)
    print("Training completed", file=sys.stdout, flush=True)
    return model

def test_model(model, dataset):
    print("Testing model", file=sys.stdout, flush=True)
    X_test = dataset['features']
    y_test = dataset['labels']
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}", file=sys.stdout, flush=True)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Evaluation complete - Loss: {loss}, Accuracy: {accuracy}", file=sys.stdout, flush=True)
    
    y_pred = model.predict(X_test)
    predicted = np.argmax(y_pred, axis=1)
    print(f"Predicted labels (first 5): {predicted[:5]}, Actual labels (first 5): {y_test[:5]}", file=sys.stdout, flush=True)
    
    metrics_bundle = {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'actual': y_test.tolist(),
        'predicted': predicted.tolist()
    }
    print("Metrics calculated", file=sys.stdout, flush=True)
    return metrics_bundle

def initialize():
    global dataset, model, test_results
    print("Initializing server", file=sys.stdout, flush=True)
    dataset = load_dataset('iris_extended_encoded.csv')
    model = build_model()
    train_model(model, dataset)
    test_results = test_model(model, dataset)
    print("Initialization completed", file=sys.stdout, flush=True)

@app.route('/iris/model/test', methods=['GET'])
def test_endpoint():
    print("API: Serving test results", file=sys.stdout, flush=True)
    if test_results is None:
        return jsonify({'error': 'No results available'}), 500
    return jsonify(test_results), 200

if __name__ == '__main__':
    initialize()
    port = 5001  # Changed from 4000 to 5001
    print(f"Server starting on http://0.0.0.0:{port}", file=sys.stdout, flush=True)
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
