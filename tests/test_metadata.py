from helpers import process_metadata
import logging
import numpy as np
from data_cleaning import get_headers, solve_mapping
from run import DATASET_DIR, METADATA_FILE
import pathlib
import json

logger = logging.getLogger(__name__)

def test_metadata():
    headers = get_headers(DATASET_DIR)

    # Should not raise any exceptions
    metadata = process_metadata(METADATA_FILE, headers)
    logger.info("Metadata processed successfully.")
    logger.info(metadata)

def test_range(tmp_path: pathlib.Path):
    x_train = np.array([110.0, np.nan, 9999.0, 168.0, 9098.0, 208.0, 105.0, 140.0, 185.0, 9110.0, 504.0, 140.0, 7777.0, 300.0, np.nan, 150.0, 9999.0, 142.0]).reshape(-1, 1)
    x_test = np.array([195.0, 150.0, 9130.0, 210.0, 160.0, 160.0, np.nan, 515.0, 170.0, 185.0, 170.0, 200.0, 193.0]).reshape(-1, 1)
    headers = ["WEIGHT2"]
    features = [{
        "id": "WEIGHT2",
        "description": "About how much do you weigh without shoes? (If respondent answers in metrics, put a 9 in the first column)[Round fractions up.]",
        "type": "continuous",
        "mapping": [
            ["7777", "NaN"],
            ["9999", "NaN"],
            ["9000-9998", "NaN"],
            ["NaN", "mean"]
        ]
    }]
    p = tmp_path / "metadata.json"
    with open(p, "w") as f:
        json.dump(features, f)

    metadata = process_metadata(str(p), headers)
    expected_metadata = [{
        "id": "WEIGHT2",
        "description": "About how much do you weigh without shoes? (If respondent answers in metrics, put a 9 in the first column)[Round fractions up.]",
        "type": "continuous",
        "mapping": [
            [7777.0, np.nan],
            [9999.0, np.nan],
            ["9000-9998", np.nan],
            [np.nan, "mean"]
        ]
    }]
    assert metadata == expected_metadata, "Metadata processing did not yield expected result"

    x_train, x_test = solve_mapping(x_train, x_test, metadata[0], 0)
    for x in (x_train, x_test):
        # All the values should be between 50 and 999 (weight in pounds)
        assert all(np.logical_and( x >= 50, x <= 999)), "Values out of range in column 5"