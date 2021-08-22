import os
import tempfile
import numpy as np
import pandas as pd
import json
import pytest
from server import create_app

df = pd.read_csv('./bank_expenses_obfuscated.csv')


@pytest.fixture
def client():
    db_fd, db_path = tempfile.mkstemp()
    app = create_app({'TESTING': True, 'DATABASE': db_path})

    with app.test_client() as client:
        yield client


def cleanup():
    if os.path.isdir('./data/'):
        os.remove('./data/log.csv')
        os.remove('./data/naive_bayes_classifier.pkl')
        os.removedirs('./data/')


def train(client, samples):
    train_df = df.iloc[samples,:]
    response = client.post('/api/sample', data=train_df.to_json(orient='records'))
    return response


def inference(client, i):
    inference_df = df.iloc[i:i + 1, :]
    expected_value = inference_df["AccountNumber"]
    inference_df = inference_df.drop(["AccountName", "AccountNumber", "AccountTypeName"], axis=1)
    dict_data = inference_df.to_dict(orient='records')[0]

    response = client.post('/api/predict', data=json.dumps(dict_data))

    return response, expected_value


def metrics(client, n):
    response = client.get("api/metrics/" + str(n))
    return response


def test_train(client):
    samples = np.random.randint(0, df.shape[0], 10000)
    response = train(client, samples)
    assert response.status_code == 200


def test_inference(client):
    response, expected_value = inference(client, 0)
    assert response.status_code == 200


def test_metric(client):
    response = metrics(client, 10000)
    cleanup()
    assert response.status_code == 200


def test_train_model(client):
    n = 100
    it = 100
    for i in range(it):
        samples = np.random.randint(0, df.shape[0], n)
        response_training = train(client, samples)
        response_metric = metrics(client, 1000)
        print(response_metric.data)
    response_metric = metrics(client, 1000)
    print("Final metrics for last 1000 samples are:",response_metric.data)

    assert response_metric.status_code == 200




