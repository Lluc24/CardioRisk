import csv
from data_cleaning import Data
from dataset import Dataset
import numpy as np
import tempfile

from methods import cross_validate
from models import LogisticRegressionGD

MAX_ITERS = 200
K_FOLD = 5
DEGREES = [1, 3, 7, 13]
GAMMAS = [0.1, 0.3, 0.7]

def run_exp3():
    print("Running Experiment 3")
    data = Data()
    data.load_from_csv("dataset", "metadata.json")
    name = tempfile.mktemp(suffix=".npz")
    data.save_to_numpy_file(name)

    for degree in DEGREES:
        with open(f"exp3_degree_{degree}.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Gamma", "Training Loss", "Validation Loss", "Weights Norm", "F1 Score", "Accuracy", "Threshold"])

            data = Data()
            data.load_from_numpy_file(name)
            data.feature_expansion(degree)
            data.add_intercept()
            data.y_train = np.where(data.y_train == -1.0, 0.0, 1.0)
            dataset = Dataset(data.x_train, data.y_train, data.num_cont_features)

            print(f"Starting {K_FOLD}-fold cross-validation")

            for gamma in GAMMAS:
                model = LogisticRegressionGD(max_iters=MAX_ITERS, gamma=gamma)
                metrics = cross_validate(model, dataset, add_weights=True, search_threshold_iterations=100)

                tr_loss = np.mean(metrics["Train Loss"]).item()
                val_loss = np.mean(metrics["Validation Loss"]).item()
                w_norm = np.mean(metrics["Weights"], axis=(1, 0)).item()
                acc = np.mean(metrics["Accuracy"]).item()
                f1 = np.mean(metrics["F1 Score"]).item()
                threshold = np.mean(metrics["Thresholds"]).item()

                writer.writerow([gamma, tr_loss, val_loss, w_norm, f1, acc, threshold])

if __name__ == "__main__":
    run_exp3()