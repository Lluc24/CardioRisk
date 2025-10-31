from data_cleaning import Data
from dataset import Dataset
from methods import cross_validate
import csv
import numpy as np
from models import MSEGradientDescent

MAX_ITERS = 200
GAMMA = 0.001
DEGREES = [1, 2, 3, 4, 7, 13]

def exp():
   with open(f"feature_expansion.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Degree", "Training Loss", "Validation Loss", "F1 Score", "Accuracy", "Threshold", "Weights Norm"])
        for degree in DEGREES:
            print(f"Running Experiment Feature Expansion Degree={degree}")
            data = Data()
            data.load_from_numpy_file("cleaned_data.npz")
            data.feature_expansion(degree=degree)
            data.add_intercept()
            dataset = Dataset(data.x_train, data.y_train, data.num_cont_features)

            model = MSEGradientDescent(max_iters=MAX_ITERS, gamma=GAMMA)
            metrics = cross_validate(model, dataset, search_threshold_iterations=100)
            writer.writerow([
                degree,
                np.mean(metrics["Train Loss"]).item(),
                np.mean(metrics["Validation Loss"]).item(),
                np.mean(metrics["F1 Score"]).item(),
                np.mean(metrics["Accuracy"]).item(),
                np.mean(metrics["Thresholds"]).item(),
                np.linalg.norm(np.mean(metrics["Weights"], axis=1)).item()
            ])

if __name__ == "__main__":
    exp()