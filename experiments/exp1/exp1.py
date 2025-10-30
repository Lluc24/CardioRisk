from collections import defaultdict
import csv
from data_cleaning import Data
from implementations import compute_gradient, mean_squared_error
from dataset import Dataset
import numpy as np

MAX_ITERS = 200

def run_exp1():
    print("Running Experiment 1")
    data = Data()
    data.load_from_csv("dataset", "metadata.json")
    data.add_intercept()
    dataset = Dataset(data.x_train, data.y_train, data.num_cont_features)

    for gamma in [0.001, 0.005, 0.01]:

        m = [defaultdict(float) for _ in range(MAX_ITERS)]

        with open(f"gamma_{gamma}.csv", "w", ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Iteration", "Training Loss", "Validation Loss", "Weights Norm"])

            # Iterate through each fold
            for k, (x_tr, y_tr, x_val, y_val, _, _) in enumerate(dataset.k_fold_generator()):
                w = np.zeros(x_tr.shape[1])
                for i in range(MAX_ITERS):
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