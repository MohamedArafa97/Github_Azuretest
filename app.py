import os

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for,request, jsonify)
import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)

PROCESSED_PATH = 'processed_data_no_outliers.csv'
MODEL_PATH ="ensemble_model.pkl"
SCALER_PATH ="scaler.pkl"
ENCODER_PATH = "encoder.pkl"


# Load the model, scaler, and encoder
ensemble_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

raw_df=pd.read_csv(PROCESSED_PATH)
counts=raw_df["zipcode"].value_counts().reset_index()
zipcodes=np.array(counts["index"])


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html', zipcodes=zipcodes)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate input
        required_features = ['bedrooms', 'bathrooms', 'm2_living', 'm2_lot', 'm2_above', 'm2_basement',
                             'floors', 'view', 'condition', 'grade', 'built_age', 'renovation_age',
                             'zipcode', 'waterfront']
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Extract features from user input
        input_data = pd.DataFrame({feature: [data[feature]] for feature in required_features})

        # Convert the input data to the correct types
        numeric_features = ['bedrooms', 'bathrooms', 'm2_living', 'm2_lot', 'm2_above', 'm2_basement',
                            'floors', 'view', 'condition', 'grade', 'built_age', 'renovation_age', 'zipcode']
        for feature in numeric_features:
            input_data[feature] = pd.to_numeric(input_data[feature], errors='coerce')
            if input_data[feature].isnull().any():
                return jsonify({'error': f'Invalid value for feature: {feature}'}), 400

        # Debugging information: Check the initial input data
        print(f"Initial input data: {input_data}")

        # Bin 'built_age' and 'renovation_age'
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        labels = list(range(len(bins) - 1))
        input_data['built_age_binned'] = pd.cut(input_data['built_age'], bins=bins, labels=labels, right=False)
        input_data['renovation_age_binned'] = pd.cut(input_data['renovation_age'], bins=bins, labels=labels, right=False)

        # Drop original columns and one-hot encode binned and categorical features
        input_data = input_data.drop(columns=['built_age', 'renovation_age'])
        input_data = pd.get_dummies(input_data, columns=['built_age_binned', 'renovation_age_binned', 'zipcode'], drop_first=True)

        # Debugging information: Check the data after binning and encoding
        print(f"Data after binning and encoding: {input_data}")

        # Compute 'living_to_lot_ratio'
        input_data['living_to_lot_ratio'] = input_data['m2_living'] / input_data['m2_lot']

        # Apply scaling to numerical features
        numerical_features = ['bedrooms', 'bathrooms', 'm2_living', 'm2_lot', 'm2_above', 'm2_basement', 'floors', 'living_to_lot_ratio']
        ordinal_features = ['condition', 'grade', 'view']
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
        input_data[ordinal_features] = encoder.transform(input_data[ordinal_features])

        # Debugging information: Check the data after scaling and encoding
        print(f"Data after scaling and encoding: {input_data}")

        # One-hot encode the waterfront feature
        input_data = pd.get_dummies(input_data, columns=['waterfront'], drop_first=True)

        # Ensure the input data has the same columns as the training data
        required_columns = ensemble_model.estimators_[0].feature_names_in_
        missing_columns = set(required_columns) - set(input_data.columns)
        for col in missing_columns:
            input_data[col] = 0
        input_data = input_data[required_columns]

        # Debugging information: Check the final processed data
        print(f"Final processed input data: {input_data}")

        # Predict the house price
        prediction = ensemble_model.predict(input_data)[0]

        # Debugging information: Check the prediction
        print(f"Prediction: {prediction}")

        # Return the prediction as JSON
        return jsonify({'predicted_price': prediction})

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print detailed error stack trace
        return jsonify({'error': str(e)}), 500
    
    

if __name__ == '__main__':
   app.run()
