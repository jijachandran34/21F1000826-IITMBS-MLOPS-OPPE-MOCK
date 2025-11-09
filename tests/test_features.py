import pandas as pd
from scripts.process_data import engineer_features

def test_petal_area_computation():
    df = pd.DataFrame({
        "sepal_length": [5.0],
        "sepal_width": [3.5],
        "petal_length": [1.4],
        "petal_width": [0.2],
        "species": ["setosa"]
    })
    result = engineer_features(df)
    assert abs(result["petal_area"].iloc[0] - 0.28) < 1e-6

def test_sepal_area_computation():
    df = pd.DataFrame({
        "sepal_length": [5.0],
        "sepal_width": [3.5],
        "petal_length": [1.4],
        "petal_width": [0.2],
        "species": ["setosa"]
    })
    result = engineer_features(df)
    assert abs(result["sepal_area"].iloc[0] - 17.5) < 1e-6

def test_petal_sepal_ratio_computation():
    df = pd.DataFrame({
        "sepal_length": [5.0],
        "sepal_width": [3.5],
        "petal_length": [1.4],
        "petal_width": [0.2],
        "species": ["setosa"]
    })
    result = engineer_features(df)
    ratio = result["petal_area"].iloc[0] / result["sepal_area"].iloc[0]
    assert abs(result["petal_sepal_ratio"].iloc[0] - ratio) < 1e-6
