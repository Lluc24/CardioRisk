from collections import defaultdict
import csv
from data_cleaning import Data
from implementations import compute_gradient, mean_squared_error
from dataset import Dataset
import numpy as np

from methods import cross_validate
from models import RegularizedLogisticRegressionGD

MAX_ITERS = 200
GAMMA = 0.7
LAMBDAS = [0.0000000001, 0.000001, 0.001, 0.1]
DEGREE = 3

def run_exp4():
    print("Running Experiment 4")
    data = Data()
    data.load_from_csv("dataset", "metadata.json")
    data.feature_expansion(DEGREE)
    data.add_intercept()
    dataset = Dataset(data.x_train, data.y_train, data.num_cont_features)

    with open(f"exp4.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Lambda", "Training Loss", "Validation Loss", "Weights Norm", "F1 Score", "Accuracy", "Threshold"])

        for lambda_ in LAMBDAS:
            model = RegularizedLogisticRegressionGD(max_iters=MAX_ITERS, gamma=GAMMA, lambda_=lambda_)
            metrics = cross_validate(model, dataset, add_weights=True, search_threshold_iterations=100)

            tr_loss = np.mean(metrics["Train Loss"]).item()
            val_loss = np.mean(metrics["Validation Loss"]).item()
            w_norm = np.mean(metrics["Weights"], axis=(1, 0)).item()
            acc = np.mean(metrics["Accuracy"]).item()
            f1 = np.mean(metrics["F1 Score"]).item()
            threshold = np.mean(metrics["Thresholds"]).item()

            writer.writerow([lambda_, tr_loss, val_loss, w_norm, f1, acc, threshold])

if __name__ == "__main__":
    run_exp4()