from dataset import Dataset
from models import *
from data_cleaning import Data
from methods import cross_validate


def main():
    data = Data()
    data.load_from_csv("dataset", "metadata.json")
    # data.feature_expansion(degree=2)
    data.add_intercept()
    dataset = Dataset(data.x_train, data.y_train, data.x_test)
    model = LogisticRegressionGD(max_iters=300)
    metrics = cross_validate(model, dataset)
    print(metrics)
    # y_te_pred = predict_labels_logistic(x_te, w, threshold=THRESHOLD)
    # y_te_pred = np.where(y_te_pred == 0, -1, 1)
    # create_csv_submission(test_ids, y_te_pred, prefix=run_name)


if __name__ == '__main__':
    main()
