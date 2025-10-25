# Evaluation Pipeline
# input - array of 0 and 1s evaluated based on the split out training set
# output - mean square error loss function, accuracy, true positive rate, true negative rate, f1 score
import numpy as np

def accuracy(y_true,y_predicted):
    """ accuracy = #correct_predictions/total_predictions """ 
    corrects=np.where(y_true == y_predicted,1,0)
    corrects_num=np.sum(corrects)
    total_num=np.sum(np.ones(y_true.shape))
    output=corrects_num/total_num    
    return output

def precision(y_true,y_predicted):
    """ precision = (FP)/(TP+FP) """
    TP=np.sum(np.where((y_predicted==1) & (y_true==1),1,0))
    FP=np.sum(np.where((y_predicted==1) & (y_true==-1),1,0))
    output=TP/(TP+FP)
    return output

def tpr(y_true,y_predicted):
    """ true positive rate = (TP)/(TP+FN) (aka recall) """ 
    TP=np.sum(np.where((y_predicted==1) & (y_true==1),1,0))
    FN=np.sum(np.where((y_predicted==-1) & (y_true==1),1,0))
    output=TP/(TP+FN)
    print(f"tpr -> TP: {TP}, FN: {FN}, TPR: {output}")
    return output 

def fpr(y_true,y_predicted):
    """ false positive rate = (FP)/(FP+TN) """
    FP=np.sum(np.where((y_predicted==1) & (y_true==-1),1,0))
    TN=np.sum(np.where((y_predicted==-1) & (y_true==-1),1,0))
    output = FP / (FP + TN)
    print(f"fpr -> FP: {FP}, TN: {TN}, FPR: {output}")
    return output

def f1(y_true,y_predicted):
    """ f1 = 2* (precision * recall) / (precision + recall) """
    prec=precision(y_true,y_predicted)
    rec=tpr(y_true,y_predicted)
    output=2*(prec*rec)/(prec+rec)
    output = 0 if np.isnan(output) else output
    print(f"f1 -> Precision: {prec}, Recall: {rec}, F1: {output}")
    return output

def metrics_summary(y_true,y_predicted):
    acc=accuracy(y_true,y_predicted)
    prec=precision(y_true,y_predicted)
    tprate=tpr(y_true,y_predicted)
    fprate=fpr(y_true,y_predicted)
    f1score=f1(y_true,y_predicted)
    return acc,prec,tprate,fprate,f1score

def metrics(y_true, y_predicted):
    n = len(y_true)
    ones_mask = y_true == 1
    tp = np.sum(y_predicted[ones_mask] == 1).item()
    fn = np.sum(y_predicted[ones_mask] == 0).item()
    tn = np.sum(y_predicted[~ones_mask] == 0).item()
    fp = np.sum(y_predicted[~ones_mask] == 1).item()

    accuracy = (tp + tn) / n  # proportion of total predictions that were correct
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # positive cases that were correctly identified
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # negative cases that were incorrectly classified as positive
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # positive predictions that were actually correct
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # harmonic mean of precision and recall
    return accuracy, recall, fpr, precision, f1
