import os
import dill
import json
import pandas as pd
import numpy as np

from sklearn import metrics as sk_metrics
from server.model.utils import init_model
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, jsonify,make_response
)

api_bp = Blueprint('api_bp', __name__, url_prefix='/api')

data_filepath = "data/log.csv"
model_path = "data/naive_bayes_classifier.pkl"


@api_bp.route("/sample", methods=["POST"])
def sample():
    if request.method == "POST":
        try:
            json_object = json.loads(request.data)
        except ValueError:
            return make_response("No inference model.", 400)

        df = pd.DataFrame(json_object)

        if os.path.isfile(model_path):
            current_model = dill.load(open(model_path, 'rb'))
            y_pred = current_model.predict(df)
            df["AccountNumberPredicted"] = y_pred

            current_model = init_model()
            full_df = pd.concat([pd.read_csv(data_filepath), df])
            y = full_df["AccountNumber"]
            current_model.fit(full_df, y)

        else:
            current_model = init_model()
            y = df["AccountNumber"]
            current_model.fit(df, y)
            y_pred = current_model.predict(df)
            df["AccountNumberPredicted"] = y_pred

        if os.path.isfile(data_filepath):
            df.to_csv(data_filepath, mode='a', header=False)
        else:
            os.mkdir('./data/')
            df.to_csv(data_filepath, mode='a', header=True)

        dill.dump(current_model, open(model_path, 'wb'))

        return make_response("Successfully updated model", 200)


@api_bp.route("/predict", methods=["POST"])
def predict():

    if os.path.isfile(model_path):
        current_model = dill.load(open(model_path, 'rb'))
    else:
        return make_response("No inference model.", 400)
    try:
        json_object = json.loads(request.data)
    except:
        return make_response("Invalid json.", 400)

    y = current_model.predict(pd.DataFrame([json_object]))[0]
    return json.dumps([str(y)])


@api_bp.route("/metrics/<int:n>", methods=["GET"])
def metrics(n):
    if os.path.isfile(data_filepath):
        number_of_current_rows = sum(1 for line in open(data_filepath)) - 1
        if 0 < n <= number_of_current_rows:
            df = pd.read_csv(data_filepath,
                             skiprows=np.linspace(1, number_of_current_rows - n + 1,
                                                  number_of_current_rows - n + 1).astype(int))
            y_pred = df["AccountNumberPredicted"]
            y_true = df["AccountNumber"]
            precision = sk_metrics.precision_score(y_true=y_true, y_pred=y_pred, zero_division=0, average="weighted")
            recall = sk_metrics.recall_score(y_true=y_true, y_pred=y_pred, zero_division=0, average="weighted")

            response_dict = {"Precision": "{:.4f}".format(precision), "Recall": "{:.4f}".format(recall)}
            return json.dumps(response_dict)
        elif 0 < n > number_of_current_rows:
            return make_response('Not enough data points', 400)
        else:
            return make_response("Model has yet not been trained.", 400)
    else:
        return make_response("Model has yet not been trained.", 400)
