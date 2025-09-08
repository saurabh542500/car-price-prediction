from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Prepare dictionary for company -> models mapping
car_models_dict = car.groupby('company')['name'].apply(list).to_dict()

@app.route('/', methods=['GET'])
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')  # Default option

    return render_template(
        'index.html',
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        car_models_json=car_models_dict
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        # Make prediction
        input_df = pd.DataFrame(
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
        )

        prediction = model.predict(input_df)

        return str(round(prediction[0], 2))

    except Exception as e:
        print("Error:", e)
        return "Error predicting price"

if __name__ == '__main__':
    app.run(debug=True)
