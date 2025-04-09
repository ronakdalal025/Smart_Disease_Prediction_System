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


# Download model from Google Drive if not present
model_url = 'https://drive.google.com/uc?id=1c-6Quw0_aw-vecIEUmKr1B8UUvivI_FB'  # Replace with your ID
model = 'rnd_forest.joblib'

if not os.path.exists(model):
    gdown.download(model_url, model, quiet=False)

# Load your model
model = joblib.load(model)

def predd(S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17):
    psymptoms = [S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17]
    #print(psymptoms)
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if isinstance(psymptoms[j], str) and psymptoms[j].lower() == a[k].lower():
                psymptoms[j]=b[k]
    psy = [psymptoms]
    pred2 = model.predict(psy)
    disp= discrp[discrp['Disease']==pred2[0]]
    disease = pred2[0]
    description = disp.values[0][1]
    recomnd = prec[prec['Disease']==pred2[0]]
    c=np.where(prec['Disease']==pred2[0])[0][0]
    precautions=[]
    for i in range(1,len(prec.iloc[c])):
          precautions.append(prec.iloc[c,i])
   
    
    return disease, description, precautions


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.get_json()
        user_symptom = data['symptoms']  # Expecting a list of features

        padded_input = user_symptom + [0] * (17-(len(user_symptom)))
        disease,description,precautions = predd(*padded_input)

        clean_precautions = [p if isinstance(p, str) and p == p else "" for p in precautions]
        clean_description = description if isinstance(description, str) and description == description else ""

        return jsonify({'prediction': disease,
                    'description': clean_description,
                    'precautions': clean_precautions})

    except Exception as e:
        import traceback
        print("Error during prediction:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
