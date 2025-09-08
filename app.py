from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Absolute paths (important for PythonAnywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "LinearRegressionModel.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Cleaned_Car_data.csv")

# Load model and dataset
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

car = pd.read_csv(DATA_PATH)

# Prepare dictionary for company -> models mapping
car_models_dict = car.groupby('company')['name'].apply(list).to_dict()

import json

@app.route('/', methods=['GET'])
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')

    return render_template(
        'index.html',
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        car_models_json=json.dumps(car_models_dict)  # ✅ Ensure JSON format
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        # Construct input DataFrame
        input_df = pd.DataFrame(
            [[car_model, company, year, driven, fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )

        # Predict
        prediction = model.predict(input_df)
        return str(round(prediction[0], 2))

    except Exception as e:
        # Log actual error for debugging
        return f"Prediction error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
