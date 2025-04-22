import os 
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
import joblib
import numpy as np
import pandas as pd

df1 = pd.read_csv('Symptom-severity.csv')   
df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
discrp = pd.read_csv('symptom_Description.csv')
prec = pd.read_csv('symptom_precaution.csv')


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Backend is running ✅"


# Download model from Google Drive if not present
model_url = 'https://drive.google.com/uc?id=1c-6Quw0_aw-vecIEUmKr1B8UUvivI_FB'  # Replace with your ID
model = 'rnd_forest.joblib'

if not os.path.exists(model):
    gdown.download(model_url, model, quiet=False)

# Load your model
model = joblib.load(model)

def predd(x, *symptoms):
    input_symptoms = list(symptoms)
    symptom_map = dict(zip(df1["Symptom"], df1["weight"]))
    weighted_input = [symptom_map.get(sym, 0) for sym in input_symptoms]
    
    num_nonzero = sum(1 for w in weighted_input if w > 0)
    dampening_factor = min(1.0, num_nonzero / 17.0)
    raw_probs = x.predict_proba([weighted_input])[0]
    adjusted_probs = raw_probs * dampening_factor
    if adjusted_probs.sum() > 0:
        adjusted_probs /= adjusted_probs.sum()

    top_n = 3
    top_indices = np.argsort(adjusted_probs)[::-1][:top_n]
    diseases = x.classes_

    results = []
    for i in top_indices:
        disease = diseases[i]
        confidence = round(adjusted_probs[i] * 100, 2)

        desc_row = discrp[discrp['Disease'] == disease]
        desc = desc_row['Description'].values[0] if not desc_row.empty else "No description available."

        prec_row = prec[prec['Disease'] == disease]
        precautions = list(prec_row.values[0][1:]) if not prec_row.empty else []

        results.append({
            "disease": disease,
            "confidence": confidence,
            "description": desc,
            "precautions": [p for p in precautions if str(p).strip().lower() != 'nan']
        })

    return results



@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])

        # Ensure exactly 17 symptoms (pad with "0" or any default)
        symptoms += ["0"] * (17 - len(symptoms))
        symptoms = symptoms[:17]

        results = predd(model, *symptoms)
        return jsonify(results)

    except Exception as e:
        import traceback
        print("Error during prediction:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
