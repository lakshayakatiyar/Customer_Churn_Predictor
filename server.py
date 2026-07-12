from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os
import traceback

app = Flask(__name__, static_folder='frontend', static_url_path='')

# Load the model with diagnostics
model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')

print("\n" + "="*60)
print("MODEL LOADING DIAGNOSTICS")
print("="*60)
print(f"Model path: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print(f"File size: {os.path.getsize(model_path)} bytes")
    model = joblib.load(model_path)
    print(f"Model type: {type(model).__name__}")
    print(f"Model loaded successfully!")
    if hasattr(model, 'feature_importances_'):
        print(f"Feature importances available: {len(model.feature_importances_)} features")
        print(f"Model has trained weights ✓")
    print("="*60 + "\n")
else:
    print(f"ERROR: Model file not found at {model_path}")
    print("Please ensure churn_model.pkl exists in the project root")
    print("="*60 + "\n")
    model = None

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Saved model file not found.'}), 500
    
    try:
        data = request.get_json(force=True)
        # Validate required fields
        required = ['tenure', 'monthlyCharges', 'contract', 'internetService', 'techSupport']
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

        tenure = int(data['tenure'])
        monthly_charges = float(data['monthlyCharges'])
        contract = data['contract']
        internet_service = data['internetService']
        tech_support = data['techSupport']

        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'Contract_One year': [1 if contract == "One year" else 0],
            'Contract_Two year': [1 if contract == "Two year" else 0],
            'InternetService_Fiber optic': [1 if internet_service == "Fiber optic" else 0],
            'InternetService_No': [1 if internet_service == "No" else 0],
            'TechSupport_No internet service': [1 if tech_support == "No internet service" else 0],
            'TechSupport_Yes': [1 if tech_support == "Yes" else 0]
        })

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        prediction_value = int(prediction[0])
        churn_prob = float(round(float(probability[0][1]) * 100, 2))
        stay_prob = float(round(float(probability[0][0]) * 100, 2))

        return jsonify({
            'prediction': prediction_value,
            'churnProbability': churn_prob,
            'stayProbability': stay_prob
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
