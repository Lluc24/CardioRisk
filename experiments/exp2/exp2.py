from collections import defaultdict
import csv
from data_cleaning import Data
from implementations import compute_gradient, mean_squared_error, logistic_gradient, logistic_loss
from dataset import Dataset
import numpy as np

MAX_ITERS = 200
K_FOLD = 5
DEGREE = 3

def run_exp1():
    print("Running Experiment 2")
    data = Data()
    data.load_from_numpy_file("cleaned_data.npz")
    data.feature_expansion(DEGREE)
    data.add_intercept()
    data.y_train = np.where(data.y_train == -1.0, 0.0, 1.0)
    dataset = Dataset(data.x_train, data.y_train, data.num_cont_features)

    print(f"Starting {K_FOLD}-fold cross-validation")

    for gamma in [0.1, 0.3]:

        m = [defaultdict(float) for _ in range(200)]

        with open(f"gamma_{gamma}.csv", "w", ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Iteration", "Training Loss", "Validation Loss", "Weights Norm"])

            # Iterate through each fold
            for k, (x_tr, y_tr, x_val, y_val, _, _) in enumerate(dataset.k_fold_generator(K_FOLD)):
                w = np.zeros(x_tr.shape[1])
                for i in range(MAX_ITERS):
                    gradient: np.ndarray = logistic_gradient(y_tr, x_tr, w)
                    w = w - gamma * gradient
                    m[i]["Weights Norm"] += np.linalg.norm(w).item()
                    m[i]["Training Loss"] += logistic_loss(y_tr, x_tr, w).item()
                    m[i]["Validation Loss"] += logistic_loss(y_val, x_val, w).item()
                print(f"Fold {k+1} finished for {gamma = }")

            for i, d in enumerate(m):
                writer.writerow([i, d["Training Loss"]/K_FOLD, d["Validation Loss"]/K_FOLD, d["Weights Norm"]/K_FOLD])

if __name__ == "__main__":
    run_exp1()