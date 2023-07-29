import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("error", InconsistentVersionWarning)

try:
    # Open classifier in global scope
    with open("models/wine.pkl", "rb") as file:
        clf = pickle.load(file)
except InconsistentVersionWarning as w:
   print(w.original_sklearn_version)

# Hi, I'm putting this commet to triger the action and run a job after my push in github.


app = FastAPI(title="Predicting Wine Class with batching")




class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]


@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}
