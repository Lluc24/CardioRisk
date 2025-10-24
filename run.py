from typing import Any

from evaluation import metrics_summary
from implementations import predict_labels_logistic, reg_logistic_regression, build_poly
from data_cleaning import load_arrays, cleaning_pipeline, save_arrays
import numpy as np
from helpers import create_csv_submission
import logging
import sys
import wandb
import json
from split import split_data

DATASET_DIR = "dataset"
METADATA_FILE = "metadata.json"

logger = logging.getLogger(__name__)

def setup_logger():
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler("logging.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Stream (stdout) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # A logging.Handler that forwards logs to wandb when a run is active
    class WandbLoggingHandler(logging.Handler):
        def __init__(self, level=logging.INFO):
            super().__init__(level)

        def emit(self, record: logging.LogRecord) -> None:
            if wandb.run:
                msg = self.format(record)
                msg_dict = json.loads(msg)
                wandb.log(msg_dict)

    wandb_handler = WandbLoggingHandler()
    wandb_handler.setLevel(logging.INFO)

    # Configure the root logger and set level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicated handlers if script is run multiple times: reset and add our handlers
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(wandb_handler)

    logger.info("Logger is set up.")


def get_run_method(*, x_train, y_train, x_test, y_test, num_cont_features):
    def run(config=None):
        with wandb.init(config):
            config = wandb.config
            lambda_ = config.lambda_
            max_iters = config.max_iters
            gamma = config.gamma
            degree = config.degree

            updated = []
            for array in [x_train, x_test]:
                intercept = np.ones((array.shape[0], 1))
                poly = build_poly(array[:, :num_cont_features], degree)
                cat = array[:, num_cont_features:]
                new_array = np.hstack([intercept, poly, cat])
                updated.append(new_array)
            new_x_train, new_x_test = updated

            initial_w = np.zeros(new_x_train.shape[1])
            w, loss = reg_logistic_regression(y=y_train, tx=new_x_train, lambda_=lambda_, initial_w=initial_w, max_iters=max_iters, gamma=gamma)
            y_test_pred = predict_labels_logistic(new_x_test, w)
            acc, prec, tprate, fprate, f1score = metrics_summary(y_test, y_test_pred)
            logger.info(json.dumps({"accuracy": acc}))
            logger.info(json.dumps({"precision": prec}))
            logger.info(json.dumps({"true_positive_rate": tprate}))
            logger.info(json.dumps({"false_positive_rate": fprate}))
            logger.info(json.dumps({"f1_score": f1score}))
            logger.info(json.dumps({"zeros": np.sum(y_test_pred == 0).item()}))
            logger.info(json.dumps({"ones": np.sum(y_test_pred == 1).item()}))
            # create_csv_submission(test_ids, y_test_pred)
        return w, loss
    return run

if __name__ == '__main__':
    setup_logger()
    wandb.login()
    with open("wandb_config.json") as f:
        sweep_config = json.load(f)
    sweep_id = wandb.sweep(sweep_config, project="ML_Project1")
    x_train, x_test, num_cont_features, y_train, train_ids, test_ids = load_arrays()
    y_train = np.where(y_train == -1, 0, 1)
    x_tr, x_te, y_tr, y_te = split_data(x_train, y_train, ratio=0.8, seed=1)
    run = get_run_method(x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te, num_cont_features=num_cont_features)
    wandb.agent(sweep_id, run, count=30)
