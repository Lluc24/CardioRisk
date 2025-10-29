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
    return data.x_train, data.y_train, data.x_test, data.test_ids

def main():
    # load_csv_and_save()  # Only need to run once to create cleaned_data.npz
    x_train, y_train, x_test, test_ids = load_numpy_file()
    dataset = Dataset(x_train, y_train, x_test=x_test)

    model = LogisticRegressionGD(max_iters=300)
    metrics = cross_validate(model, dataset)

    # Average weights from cross-validation
    weights = np.mean(metrics.pop('Weights'), axis=0)

    # Predict on test set
    model.w = weights  # Set model weights
    y_te_pred = model.predict(x_test)

    # Create submission file
    create_csv_submission(test_ids, y_te_pred)

    print(metrics)


if __name__ == '__main__':
    main()
