import pytest
import numpy as np
from data_cleaning import clean_irrelevant_features, clean_few_values_features, clean_categorical_features, clean_continuous_features

def test_clean_irrelevant_features():
    x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    headers = ["feature1", "feature2", "feature3", "feature4"]
    irrelevant_features = [
        {
            "id": "feature2",
            "description": "X"
        },
        {
            "id": "feature4",
            "description": "Y"
        }
    ]

    expected_x = np.array([
        [1, 3],
        [5, 7],
        [9, 11]
    ])

    cleaned_x = clean_irrelevant_features(x, headers, irrelevant_features)
    assert np.array_equal(cleaned_x, expected_x), "The irrelevant features were not removed correctly."

def test_clean_few_values_features():
    x = np.array([
        [1, 2, 3, 4],
        [1, np.nan, 3, 5],
        [1, 2, 3, 6],
        [1, np.nan, 3, 5],
    ])
    headers = ["feature1", "feature2", "feature3", "feature4"]
    few_values_features = [
        {
            "id": "feature2",
            "description": "X",
            "values": 2
        }
    ]
    few_values_features_error = [
        {
            "id": "feature2",
            "description": "X",
            "values": 3
        }
    ]

    expected_x = np.array([
        [1, 3, 4],
        [1, 3, 5],
        [1, 3, 6],
        [1, 3, 5],
    ])

    with pytest.raises(ValueError):
        clean_few_values_features(x, headers, few_values_features_error)

    cleaned_x = clean_few_values_features(x, headers, few_values_features)
    assert np.array_equal(cleaned_x, expected_x), "The features with few unique values were not removed correctly."

def test_clean_categorical_features():
    x = np.array([
        [7, 0],
        [np.nan, 1],
        [2, 0],
        [8, 1],
    ])
    headers = ["feature1", "feature2", "feature3", "feature4"]
    categorical_features = [
        {
            "id": "feature1",
            "description": "X",
            "mapping": {
                "NaN": 0,
                "8": 1,
                "7": "NaN"
            }
        }
    ]
    expected_x = np.array([
        [0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
    ])

    cleaned_x = clean_categorical_features(x, headers, categorical_features)
    assert np.array_equal(cleaned_x, expected_x), "The categorical features were not encoded correctly."

def test_continuous_features():
    x = np.array([
        [77, 99, 3],
        [5, 1, np.nan],
        [2, 80, 5],
        [3, np.nan, 8],
    ])
    headers = ["feature1", "feature2", "feature3"]
    continuous_features = [
        {
            "id": "feature1",
            "description": "X",
            "mapping": {
                "80": 0,
                "77": "NaN",
                "99": "NaN"
            }
        },
        {
            "id": "feature2",
            "description": "Y",
            "mapping": {
                "80": 0,
                "77": "NaN",
                "99": "NaN"
            }
        }
    ]
    expected_x = np.array([
        [np.nan, np.nan, 3],
        [5, 1, np.nan],
        [2, 0, 5],
        [3, np.nan, 8],
    ])

    cleaned_x = clean_continuous_features(x, headers, continuous_features)
    assert np.array_equal(cleaned_x, expected_x, equal_nan=True), "The continuous features were not cleaned correctly."
