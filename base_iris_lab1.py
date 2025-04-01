# base_iris_lab1.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Global lists to store datasets, models, and metrics
datasets = []
models = []
metrics = []

def load_local(filepath='iris_extended_encoded.csv'):
    """
    Load the dataset from a CSV file and preprocess it.
    """
    df = pd.read_csv(filepath, header=0)
    features = df.iloc[:, 1:21].values
    labels = df.iloc[:, 0].values
    
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    dataset = {
        'dataframe': df,
        'features': features,
        'labels': encoded_labels,
        'label_encoder': le
    }
    datasets.append(dataset)
    return len(datasets) - 1

def build():
    """
    Build a neural network model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=20),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    models.append(model)
    return len(models) - 1

def train(model_idx, dataset_idx):
    """
    Train the model on the specified dataset.
    """
    model = models[model_idx]
    dataset = datasets[dataset_idx]
    history = model.fit(
        dataset['features'],
        dataset['labels'],
        epochs=50,
        validation_split=0.2
    )
    return history.history

def score(model_idx, features):
    """
    Score the model with the provided features.
    """
    model = models[model_idx]
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    dataset = datasets[0]  # Assuming using first dataset's encoder
    return dataset['label_encoder'].inverse_transform([np.argmax(prediction)])[0]

def new_model(dataset_idx):
    """
    Build and train a new model using the specified dataset.
    """
    model_idx = build()
    train(model_idx, dataset_idx)
    return model_idx

def save_metrics(model_id, metrics_bundle):
    """
    Save metrics for a model.
    """
    if model_id < len(metrics):
        metrics[model_id] = metrics_bundle
    else:
        metrics.append(metrics_bundle)
    return

def test(model_id, dataset_id):
    """
    Test a model with a dataset and return metrics.
    """
    model = models[model_id]
    dataset = datasets[dataset_id]
    
    # Use pre-extracted features and labels, consistent with train
    X_test = dataset['features']
    y_test = dataset['labels']
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    predicted = np.argmax(y_pred, axis=1)
    
    metrics_bundle = {
        'accuracy': float(accuracy),  # Ensure float for JSON
        'loss': float(loss),
        'actual': y_test.tolist(),
        'predicted': predicted.tolist()
    }
    
    save_metrics(model_id, metrics_bundle)
    return metrics[model_id]