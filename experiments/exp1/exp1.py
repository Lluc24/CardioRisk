from collections import defaultdict
import csv
from data_cleaning import Data
from implementations import compute_gradient, mean_squared_error
from dataset import Dataset
import numpy as np


def run_exp1():
    print("Running Experiment 1")
    data = Data()
    data.load_from_csv("dataset", "metadata.json")
    data.add_intercept()
    data.load_from_numpy_file("cleaned_data.npz")
    dataset = Dataset(data.x_train, data.y_train, data.num_cont_features)

    k_fold = 5

    print(f"Starting {k_fold}-fold cross-validation")

    for gamma in [0.001, 0.01]:

        m = [defaultdict(float) for _ in range(200)]

        with open(f"gamma_{gamma}.csv", "w", ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Iteration", "Training Loss", "Validation Loss", "Weights Norm"])

            # Iterate through each fold
            for k, (x_tr, y_tr, x_val, y_val, _, _) in enumerate(dataset.k_fold_generator(k_fold)):
                w = np.zeros(x_tr.shape[1])
                for i in range(200):
                    gradient: np.ndarray = compute_gradient(y_tr, x_tr, w)
                    w = w - gamma * gradient
                    m[i]["Weights Norm"] += np.linalg.norm(w).item()
                    m[i]["Training Loss"] += mean_squared_error(y_tr, x_tr, w).item()
                    m[i]["Validation Loss"] += mean_squared_error(y_val, x_val, w).item()

                print(f"Fold {k+1} finished for {gamma = }")

            for i, d in enumerate(m):
                writer.writerow([i, d["Training Loss"]/5, d["Validation Loss"]/5, d["Weights Norm"]/5])

if __name__ == "__main__":
    run_exp1()