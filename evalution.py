# Evaluation Pipeline
# input - array of 0 and 1s evaluated based on the split out training set
# output - mean square error loss function, accuracy, recall, true positive rate, true negative rate


def accuracy(y_true,y_predicted):
    accuracy = metrics.accuracy_score(y_true, y_predicted)

def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    """Accuracy gives number of correct predictions divided by total number of predictions.

    y_true : 1d array-like
    y_pred : 1d array-like

    """



#    fpr, tpr, threshold = metrics.roc_curve(y_true, y_predicted)
#    f1=metrics.f1_score(y_true, y_predicted)
#    return accuracy, fpr, tpr, f1