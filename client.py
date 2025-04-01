# client.py
import requests
import time

BASE_URL = "http://localhost:4000/iris"

def upload_dataset(file_path):
    """Upload a dataset to the API"""
    try:
        with open(file_path, 'rb') as f:
            files = {'train': f}
            response = requests.post(f"{BASE_URL}/datasets", files=files)
        response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Upload failed: Status {e.response.status_code if e.response else 'N/A'}")
        print(f"Response: {e.response.text if e.response else str(e)}")
        raise
    except ValueError:
        print("Upload failed: Response is not JSON")
        print(f"Response text: {response.text}")
        raise

def create_model(dataset_id):
    """Create a new model"""
    try:
        data = {'dataset': str(dataset_id)}
        response = requests.post(f"{BASE_URL}/model", data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Create model failed: Status {e.response.status_code if e.response else 'N/A'}")
        print(f"Response: {e.response.text if e.response else str(e)}")
        raise
    except ValueError:
        print("Create model failed: Response is not JSON")
        print(f"Response text: {response.text}")
        raise

def retrain_model(model_id, dataset_id):
    """Retrain an existing model"""
    try:
        params = {'dataset': str(dataset_id)}
        response = requests.put(f"{BASE_URL}/model/{model_id}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Retrain failed: Status {e.response.status_code if e.response else 'N/A'}")
        print(f"Response: {e.response.text if e.response else str(e)}")
        raise
    except ValueError:
        print("Retrain failed: Response is not JSON")
        print(f"Response text: {response.text}")
        raise

def score_model(model_id, features):
    """Score a single row with the model"""
    try:
        params = {'fields': ','.join(map(str, features))}
        response = requests.get(f"{BASE_URL}/model/{model_id}/score", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Score failed: Status {e.response.status_code if e.response else 'N/A'}")
        print(f"Response: {e.response.text if e.response else str(e)}")
        raise
    except ValueError:
        print("Score failed: Response is not JSON")
        print(f"Response text: {response.text}")
        raise

def test_model(model_id, dataset_id):
    """Test the model with a dataset"""
    try:
        params = {'dataset': str(dataset_id)}
        response = requests.get(f"{BASE_URL}/model/{model_id}/test", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Test failed: Status {e.response.status_code if e.response else 'N/A'}")
        print(f"Response: {e.response.text if e.response else str(e)}")
        raise
    except ValueError:
        print("Test failed: Response is not JSON")
        print(f"Response text: {response.text}")
        raise

def main():
    """Main driver function to test all API endpoints"""
    print("=== Starting API Test ===")
    
    # 1. Upload dataset
    print("\n1. Uploading dataset...")
    try:
        upload_response = upload_dataset('iris_extended_encoded.csv')
        dataset_id = upload_response['dataset_id']
        print(f"Uploaded dataset with ID: {dataset_id}")
    except Exception as e:
        print(f"Skipping to next step due to error: {e}")
        return
    
    # 2. Create and train a new model
    print("\n2. Creating and training a new model...")
    try:
        model_response = create_model(dataset_id)
        model_id = model_response['model_id']
        print(f"Created model with ID: {model_id}")
    except Exception as e:
        print(f"Skipping to next step due to error: {e}")
        return
    
    # Wait a moment for training to complete
    time.sleep(2)
    
    # 3. Retrain the model
    print("\n3. Retraining the model...")
    try:
        retrain_response = retrain_model(model_id, dataset_id)
        print("Retraining results:", retrain_response)
    except Exception as e:
        print(f"Skipping to next step due to error: {e}")
    
    # 4. Score a single row
    print("\n4. Scoring a single row...")
    example_features = [5.1, 3.5, 1.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    try:
        score_response = score_model(model_id, example_features)
        print(f"Predicted species: {score_response['species']}")
    except Exception as e:
        print(f"Skipping to next step due to error: {e}")
    
    # 5. Test the model with the dataset
    print("\n5. Testing the model with the dataset...")
    try:
        test_response = test_model(model_id, dataset_id)
        print("Test results:")
        print(f"Accuracy: {test_response['accuracy']}")
        print(f"Loss: {test_response['loss']}")
        print(f"First 5 actual: {test_response['actual'][:5]}")
        print(f"First 5 predicted: {test_response['predicted'][:5]}")
    except Exception as e:
        print(f"Test error: {e}")
    
    print("\n=== API Test Completed ===")

if __name__ == '__main__':
    main()