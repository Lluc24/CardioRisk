from typing import Any

from evaluation import metrics, metrics_summary
from implementations import predict_labels_logistic, reg_logistic_regression, build_poly
from data_cleaning import load_arrays, cleaning_pipeline, save_arrays, prepare_arrays
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


def get_run_method(*, x_tr, y_tr, x_te, test_ids, num_cont_features):
    def run(config=None):
        with wandb.init(config):
            run_name = wandb.run.name
            config = wandb.config
            lambda_ = config.lambda_
            max_iters = config.max_iters
            gamma = config.gamma
            degree = config.degree

            x_train, x_test = prepare_arrays(x_tr, x_te, num_cont_features, degree)
            x_train, x_validation, y_train, y_validation = split_data(x_train, y_tr, ratio=0.8, seed=2)
            w = np.zeros(x_train.shape[1])
            w, loss = reg_logistic_regression(y=y_train, tx=x_train, lambda_=lambda_, initial_w=w, max_iters=max_iters, gamma=gamma)
            y_val_pred = predict_labels_logistic(x_validation, w)
            accuracy, recall, fpr, precision, f1 = metrics(y_validation, y_val_pred)
            logger.info(json.dumps({"Ones": np.sum(y_val_pred).item()}))
            logger.info(json.dumps({"Zeros": len(y_val_pred) - np.sum(y_val_pred).item()}))
            logger.info(json.dumps({"Accuracy": accuracy}))
            logger.info(json.dumps({"Recall": recall}))
            logger.info(json.dumps({"False_Positive_Rate": fpr}))
            logger.info(json.dumps({"Precision": precision}))
            logger.info(json.dumps({"F1_Score": f1}))
            logger.info(json.dumps({"OLA_Metrics": f"{metrics_summary(y_validation, y_val_pred)}"}))
            y_test_pred = predict_labels_logistic(x_test, w)
            y_test_pred = np.where(y_test_pred == 0, -1, 1)
            create_csv_submission(test_ids, y_test_pred, prefix=run_name)
        return w, loss
    return run


def main():
    setup_logger()
    wandb.login()
    with open("wandb_config.json") as f:
        sweep_config = json.load(f)
    sweep_id = wandb.sweep(sweep_config, project="ML_Project1")
    x_train, x_test, num_cont_features, y_train, train_ids, test_ids = load_arrays()
    y_train = np.where(y_train == -1, 0, 1)
    run = get_run_method(x_tr=x_train, y_tr=y_train, x_te=x_test, test_ids=test_ids,
                         num_cont_features=num_cont_features)
    wandb.agent(sweep_id, run, count=30)

if __name__ == '__main__':
    main()
