# 1. Library imports
import uvicorn
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import shap
import json
pd.options.mode.chained_assignment = None  # default='warn'

# 2. Create the app object
app = FastAPI()
# Chargement du modèle
model = joblib.load('model.pkl')
data = joblib.load('sample_test_set.pickle')
list_ID = data.index.tolist()
# Enregistrer le model
classifier = model.named_steps['classifier']

@app.get("/predict/{client_id}")
async def predict(client_id : int):
    predictions = model.predict_proba(data).tolist()
    predict_proba = []
    for pred, ID in zip(predictions, list_ID):
        if ID == client_id:
            predict_proba.append(pred[1])
    return predict_proba[0]

@app.get('/generic_shap')
async def generic_shap():
    df_preprocess = model.named_steps['preprocessor'].transform(data)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(df_preprocess, check_additivity=False)
    shap_values_list = [value.tolist() for value in shap_values]
    json_shap = json.dumps(shap_values_list)
    return {'shap_values':json_shap}

@app.get('/shap_client/{client_id}')
async def shap_client(client_id : int):
    index_ID = []
    for ind, ID in enumerate(list_ID):
        if list_ID[ind] == client_id:
            index_ID.append(ind)
        else:
            pass
    df_preprocess = model.named_steps['preprocessor'].transform(data)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(df_preprocess, check_additivity=False)
    shap_values_client = shap_values[index_ID][0]
    json_shap_client = json.dumps(shap_values_client.tolist())
    return {'shap_client':json_shap_client}

# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='35.180.29.152', port=8000, reload=True)

