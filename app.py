# app.py
from flask import Flask, request, jsonify
from base_iris_lab1 import load_local, build, train, score, new_model, test

app = Flask(__name__)

# Global lists to store dataset and model indices
datasets = []
models = []

# Upload training data
@app.route('/iris/datasets', methods=['POST'])
def upload_dataset():
    """
    Upload a dataset via a CSV file.
    """
    if 'train' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['train']
    file.save('temp.csv')
    try:
        dataset_id = load_local('temp.csv')
        datasets.append(dataset_id)
        return jsonify({'dataset_id': dataset_id}), 201
    except Exception as e:
        return jsonify({'error': f'Failed to load dataset: {str(e)}'}), 500

# Build and train a new model
@app.route('/iris/model', methods=['POST'])
def create_model():
    """
    Create and train a new model using the specified dataset.
    """
    try:
        dataset_id = int(request.form.get('dataset'))
        if dataset_id >= len(datasets) or dataset_id < 0:
            return jsonify({'error': 'Invalid dataset_id'}), 400
        model_id = new_model(dataset_id)
        models.append(model_id)
        return jsonify({'model_id': model_id}), 201
    except ValueError:
        return jsonify({'error': 'dataset_id must be an integer'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to create model: {str(e)}'}), 500

# Retrain an existing model
@app.route('/iris/model/<int:model_id>', methods=['PUT'])
def retrain_model(model_id):
    """
    Retrain an existing model using the specified dataset.
    """
    try:
        if model_id >= len(models) or model_id < 0:
            return jsonify({'error': 'Invalid model_id'}), 400
        dataset_id = int(request.args.get('dataset'))
        if dataset_id >= len(datasets) or dataset_id < 0:
            return jsonify({'error': 'Invalid dataset_id'}), 400
        history = train(model_id, dataset_id)
        return jsonify(history), 200
    except ValueError:
        return jsonify({'error': 'dataset_id must be an integer'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to retrain model: {str(e)}'}), 500

# Score a model with provided features
@app.route('/iris/model/<int:model_id>/score', methods=['GET'])
def score_model(model_id):
    """
    Score a model using the provided features.
    """
    try:
        if model_id >= len(models) or model_id < 0:
            return jsonify({'error': 'Invalid model_id'}), 400
        fields = list(map(float, request.args.get('fields').split(',')))
        if len(fields) != 20:  # Assuming 20 features as per your base code
            return jsonify({'error': 'Expected 20 features'}), 400
        prediction = score(model_id, fields)
        return jsonify({'species': prediction}), 200
    except ValueError:
        return jsonify({'error': 'fields must be a comma-separated list of numbers'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to score model: {str(e)}'}), 500

# Test a model with a dataset
@app.route('/iris/model/<int:model_id>/test', methods=['GET'])
def test_model(model_id):
    """
    Test a model using the specified dataset.
    """
    try:
        if model_id >= len(models) or model_id < 0:
            return jsonify({'error': 'Invalid model_id'}), 400
        dataset_id = int(request.args.get('dataset'))
        if dataset_id >= len(datasets) or dataset_id < 0:
            return jsonify({'error': 'Invalid dataset_id'}), 400
        test_results = test(model_id, dataset_id)
        return jsonify(test_results), 200
    except ValueError:
        return jsonify({'error': 'dataset_id must be an integer'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to test model: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)