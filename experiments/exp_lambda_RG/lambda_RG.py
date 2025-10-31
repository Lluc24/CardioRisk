import csv
from data_cleaning import Data
from dataset import Dataset
import numpy as np

from methods import cross_validate
from models import LogisticRegressionGD

MAX_ITERS = 300
GAMMAS = np.logspace(-2, 0, 10)

def run_exp1():
    print("Running Experiment 4")
    data = Data()
    data.load_from_csv("dataset", "metadata.json")
    data.add_intercept()
    dataset = Dataset(data.x_train, data.y_train, data.num_cont_features)

    with open(f"lambda_RG.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Lambda", "F1 Score", "Accuracy", "Threshold"])

        for gamma in GAMMAS:
            model = LogisticRegressionGD(max_iters=MAX_ITERS, gamma=gamma)
            metrics = cross_validate(model, dataset, add_weights=False, search_threshold_iterations=100)

            acc = np.mean(metrics["Accuracy"]).item()
            f1 = np.mean(metrics["F1 Score"]).item()
            threshold = np.mean(metrics["Thresholds"]).item()

            writer.writerow([gamma, f1, acc, threshold])

if __name__ == "__main__":
    run_exp1()