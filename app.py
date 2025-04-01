import requests
import numpy as np

BASE_URL = "http://localhost:5001"
TEST_FILE = "iris_extended_encoded.csv"

def upload_dataset(file_path):
    print("\n1. Uploading dataset...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/iris/datasets",
                files={'file': f}
            )
        
        if response.status_code == 201:
            dataset_id = response.json()['dataset_id']
            print(f"✓ Uploaded dataset with ID: {dataset_id}")
            return dataset_id
        else:
            print(f"✗ Error uploading dataset: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Exception during upload: {str(e)}")
        return None

def create_model(dataset_id):
    print("\n2. Creating and training model...")
    try:
        response = requests.post(
            f"{BASE_URL}/iris/model",
            data={'dataset': dataset_id}
        )
        
        if response.status_code == 201:
            result = response.json()
            print(f"✓ Created model with ID: {result['model_id']}")
            print(f"   Trained on dataset ID: {result['dataset_id']}")
            return result['model_id']
        else:
            print(f"✗ Error creating model: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Exception during model creation: {str(e)}")
        return None

def retrain_model(model_id, dataset_id):
    print(f"\n3. Retraining model {model_id}...")
    try:
        response = requests.put(
            f"{BASE_URL}/iris/model/{model_id}/train",  
            params={'dataset': dataset_id}
        )
        
        if response.status_code == 200:
            print(f"✓ Retrained model {model_id} on dataset {dataset_id}")
            return True
        else:
            print(f"✗ Error retraining model: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Exception during retraining: {str(e)}")
        return False

def test_model(model_id, dataset_id):
    print(f"\n4. Testing model {model_id}...")
    try:
        response = requests.get(
            f"{BASE_URL}/iris/model/{model_id}/test",
            params={'dataset': dataset_id}
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✓ Test results for model {model_id}:")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   Loss: {results['loss']:.4f}")
            print("   First 5 predictions vs actual:")
            for i in range(5):
                print(f"     {results['predicted_labels'][i]} (pred) vs {results['actual_labels'][i]} (actual)")
            return results
        else:
            print(f"✗ Error testing model: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Exception during testing: {str(e)}")
        return None

def predict(model_id):
    print(f"\n5. Making prediction with model {model_id}...")
    try:
        # Create a realistic sample (replace with actual feature values if needed)
        sample_features = [5.1, 3.5, 1.4, 0.2] + [0.1]*16  # First 4 are real iris features
        
        response = requests.get(
            f"{BASE_URL}/iris/model/{model_id}/predict",
            params={'fields': ','.join(map(str, sample_features))}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction: {result['species']}")
            print("   Probabilities:")
            for i, prob in enumerate(result['probabilities']):
                print(f"     Class {i}: {prob:.4f}")
            return result
        else:
            print(f"✗ Error predicting: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Exception during prediction: {str(e)}")
        return None

def main():
    print("\n=== Iris Model API Client ===")
    
    # 1. Upload dataset
    dataset_id = upload_dataset(TEST_FILE)
    if dataset_id is None:
        print("\n✗ Stopping due to dataset upload failure")
        return
    
    # 2. Create and train model
    model_id = create_model(dataset_id)
    if model_id is None:
        print("\n✗ Stopping due to model creation failure")
        return
    
    # 3. Retrain model (optional)
    if not retrain_model(model_id, dataset_id):
        print("\n⚠ Continuing despite retraining failure")
    
    # 4. Test model
    test_results = test_model(model_id, dataset_id)
    if test_results is None:
        print("\n⚠ Continuing despite testing failure")
    
    # 5. Make prediction
    prediction = predict(model_id)
    if prediction is None:
        print("\n⚠ Continuing despite prediction failure")
    
    print("\n=== Operation Summary ===")
    print(f"Dataset ID used: {dataset_id}")
    print(f"Model ID used: {model_id}")
    if test_results:
        print(f"Final accuracy: {test_results['accuracy']:.4f}")
    print("=======================")

if __name__ == "__main__":
    main()
