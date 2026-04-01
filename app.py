from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

model = joblib.load('xgb_model.pkl')
ct    = joblib.load('column_transformer.pkl')


def get_india_weather():
    """
    Returns weather based on India's seasonal patterns using current month.
    Months -> Season -> Likely Weather
    """
    month = datetime.now().month

    season_weather = {
        1:  'CLOUDY',  
        2:  'SUNNY',  
        3:  'SUNNY',  
        4:  'SUNNY',  
        5:  'STORMY', 
        6:  'RAINY',  
        7:  'RAINY',   
        8:  'RAINY',   
        9:  'RAINY',  
        10: 'CLOUDY',  
        11: 'CLOUDY', 
        12: 'SUNNY', 
    }

    return season_weather[month]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required = ['pickup_prefix', 'delivery_prefix', 'product_category', 'weight_bucket']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        weather = get_india_weather()

        new_data = pd.DataFrame([{
            'pickup_prefix':    data['pickup_prefix'],
            'delivery_prefix':  data['delivery_prefix'],
            'product_category': data['product_category'],
            'weather':          weather,
            'weight_bucket':    data['weight_bucket']
        }])

        new_encoded   = ct.transform(new_data)
        proba         = model.predict_proba(new_encoded)[0]
        predicted_day = int(model.predict(new_encoded)[0]) + 1

        all_probs = {
            int(day) + 1: round(float(prob) * 100, 2)
            for day, prob in zip(model.classes_, proba)
        }

        top_3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_result = [
            {'day': day, 'probability': prob}
            for day, prob in top_3
        ]

        return jsonify({
            'predicted_day': predicted_day,
            'weather_used':  weather,
            'top_3_days':    top_3_result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'weather_this_month': get_india_weather()})


if __name__ == '__main__':
    app.run(debug=True, port=5000)