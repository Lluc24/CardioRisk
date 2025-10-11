import pytest
import numpy as np
import pathlib
from data_cleaning import get_headers, standardize, add_intercept_column, one_hot_encoding, solve_mapping, \
    cleaning_pipeline
import json

def test_add_intercept_column():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_x = np.array([[1, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9]])
    intercept_x = add_intercept_column(x)
    assert np.array_equal(intercept_x, expected_x), "The intercept column was not added correctly."

def test_standardize():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_x = np.array([[-1.22474487, -1.22474487, -1.22474487],
                           [0., 0., 0.],
                           [1.22474487, 1.22474487, 1.22474487]])
    standardized_x = standardize(x)
    assert np.allclose(standardized_x, expected_x), "The standardization did not produce the expected result."

def test_get_headers(tmp_path: pathlib.Path):
    expected_headers = ["feature1", "feature2", "feature3"]
    headers = [
        "id",
        "feature1",
        "feature2",
        "feature3"
    ]
    with open(tmp_path / "x_train.csv", "w") as f:
        f.write(",".join(headers) + "\n")
        f.write("1,2,3,4\n")
        f.write("5,6,7,8\n")

    loaded_headers = get_headers(str(tmp_path))
    assert loaded_headers == expected_headers, f"Expected headers {expected_headers}, but got {loaded_headers}"


@pytest.mark.skip
def test_solve_mapping():

    # Test case 1: Correct mapping
    feature = {
        "id": "feature1",
        "type": "continuous",
        "mapping":[
            ["NaN", "2"],
            ["7", "NaN"],
            ["9", "NaN"],
            ["NaN", "mean"]
        ],
        "vocabulary": []
    }
    x_train = np.array([
        [1, 100],
        [2, 200],
        [3, 300],
        [np.nan, 400]
    ])
    x_test = np.array([
        [2, 150],
        [3, 250],
        [1, 350],
        [np.nan, 450]
    ])
    expected_x_train = np.array([
        [10, 100],
        [20, 200],
        [30, 300],
        [-1, 400]
    ])
    expected_x_test = np.array([
        [20, 150],
        [30, 250],
        [10, 350],
        [-1, 450]
    ])
    actual_x_train, actual_x_test = solve_mapping(x_train, x_test, feature, 0)
    assert np.array_equal(actual_x_train, expected_x_train), f"Test case 1 failed for x_train. Expected {expected_x_train}, but got {actual_x_train}"
    assert np.array_equal(actual_x_test, expected_x_test), f"Test case 1 failed for x_test. Expected {expected_x_test}, but got {actual_x_test}"

    # Test case 2: Value not in mapping
    x_test[0, 0] = 4  # 4 is not in the mapping
    with pytest.raises(ValueError):
        solve_mapping(x_train, x_test, feature, 0)

@pytest.mark.skip
def test_one_hot_encoding():
    rng = np.random.default_rng()

    # Test case 1: Correct one-hot encoding
    ordered_vocabulary = ["1", "2", "3", "NaN"]
    feature = {
        "id": "feature1",
        "type": "categorical",
        "mapping": {},
        "vocabulary": ordered_vocabulary
    }
    x_train = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [np.nan, 40]
    ])
    x_test = np.array([
        [2, 15],
        [3, 25],
        [1, 35],
        [np.nan, 45]
    ])
    expected_x_train = np.array([
        [10, 1, 0, 0, 0],
        [20, 0, 1, 0, 0],
        [30, 0, 0, 1, 0],
        [40, 0, 0, 0, 1]
    ])
    expected_x_test = np.array([
        [15, 0, 1, 0, 0],
        [25, 0, 0, 1, 0],
        [35, 1, 0, 0, 0],
        [45, 0, 0, 0, 1]
    ])
    actual_x_train, actual_x_test = one_hot_encoding(x_train, x_test, feature, 0)
    assert np.array_equal(actual_x_train, expected_x_train), f"Test case 1 failed for x_train. Expected {expected_x_train}, but got {actual_x_train}"
    assert np.array_equal(actual_x_test, expected_x_test), f"Test case 1 failed for x_test. Expected {expected_x_test}, but got {actual_x_test}"

    # Test case 2: Vocabulary with duplicates
    e = rng.choice(feature["vocabulary"])
    feature["vocabulary"].append(e)
    rng.shuffle(feature["vocabulary"])
    with pytest.raises(ValueError):
        one_hot_encoding(x_train, x_test, feature, 0)

    # Test case 3: Vocabulary not sorted
    while feature["vocabulary"] == ordered_vocabulary:
        feature["vocabulary"] = list(rng.permutation(feature["vocabulary"]))
    with pytest.raises(ValueError):
        one_hot_encoding(x_train, x_test, feature, 0)

    # Test case 4: Values not in the vocabulary
    x_test[0, 0] = 4  # 4 is not in the vocabulary
    feature["vocabulary"] = ordered_vocabulary
    with pytest.raises(ValueError):
        one_hot_encoding(x_train, x_test, feature, 0)
