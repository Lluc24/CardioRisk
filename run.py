from dataset import Dataset
from helpers import create_csv_submission
from models import *
from data_cleaning import Data
from methods import cross_validate

def load_csv_and_save():
    data = Data()
    data.load_from_csv("dataset", "metadata.json")
    # data.feature_expansion(degree=2)
    data.add_intercept()
    data.save_to_numpy_file("cleaned_data.npz")

def load_numpy_file():
    data = Data()
    data.load_from_numpy_file("cleaned_data.npz")
    return data.x_train, data.y_train, data.x_test, data.test_ids, data.num_cont_features

def main():
    #load_csv_and_save()  # Only need to run once to create cleaned_data.npz
    x_train, y_train, x_test, test_ids, num_cont_features = load_numpy_file()
    dataset = Dataset(x_train, y_train, num_cont_features)

    model = LogisticRegressionGD()
    metrics = cross_validate(model, dataset, search_threshold_iterations=1000)

    # Average weights and threshold from cross-validation
    weights = np.mean(metrics.pop('Weights'), axis=0)
    threshold = float(np.mean(metrics.pop('Thresholds')))

    # Average mean and std from cross-validation
    mean = np.mean(metrics.pop('Mean'))
    std = np.mean(metrics.pop('Std'))

    # Predict on local test set
    x_test_local, y_test_local = dataset.get_test_set()
    x_test_local[:, 1:num_cont_features] = (x_test_local[:, 1:num_cont_features] - mean) / std  # Standardize test set
    y_te_local_pred = model.predict(x_test_local, threshold=threshold)
    local_metrics = model.get_metrics(y_test_local, y_te_local_pred)
    print("Local Test Set Metrics:", local_metrics)

    # Predict on online test set

    model.w = weights  # Set model weights
    x_test[:, 1:num_cont_features] = (x_test[:, 1:num_cont_features] - mean) / std  # Standardize test set
    y_te_pred = model.predict(x_test, threshold=threshold)

    # Create submission file
    create_csv_submission(test_ids, y_te_pred)

    print(metrics)


if __name__ == '__main__':
    main()
