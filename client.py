# client.py
import requests
import json

BASE_URL = "http://localhost:5001"

def test_model():
    print("5. Testing the model...")
    try:
        response = requests.get(f"{BASE_URL}/iris/model/test")
        if response.status_code == 200:
            results = response.json()
            print(f"Test successful: Accuracy: {results['accuracy']}, Loss: {results['loss']}")
            print(f"First 5 predictions: {results['predicted'][:5]}")
            print(f"First 5 actual: {results['actual'][:5]}")
            return results
        else:
            print(f"Test failed: Status {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Test error: {str(e)}")
        return None

def main():
    print("Starting client driver for Mini2 Lab2")
    # For simplicity, we only need the test endpoint since training is server-side
    test_results = test_model()
    if test_results:
        print("=== API Test Completed ===")
    else:
        print("=== API Test Failed ===")

if __name__ == "__main__":
    main()
