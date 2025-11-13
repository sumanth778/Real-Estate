from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
app = Flask(__name__)


model = joblib.load('Random_forest_price_model.pkl')


location_le = LabelEncoder()
location_le.classes_ = np.array(['Manhattan', 'Brooklyn', 'Bronx', 'Queens', 'Staten Island', 'Harlem'])

property_type_le = LabelEncoder()
property_type_le.classes_ = np.array(['Apartment', 'Condo', 'House'])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_data = {
            'bedrooms': int(data['bedrooms']),
            'bathrooms': int(data['bathrooms']),
            'square_feet': int(data['square_feet']),
            'location': location_le.transform([data['location']])[0],
            'year_built': int(data['year_built']),
            'garage': int(data['garage']),
            'has_pool': int(data['has_pool']),
            'property_type': property_type_le.transform([data['property_type']])[0],
            'num_floors': int(data['num_floors']),
            'has_basement': int(data['has_basement'])
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        return jsonify({'predicted_category': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)