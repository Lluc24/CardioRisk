# Evaluation Pipeline
# input - array of 0 and 1s evaluated based on the split out training set
# output - mean square error loss function, accuracy, true positive rate, true negative rate, f1 score


def accuracy(y_true,y_predicted):
    ' accuracy = #correct_predictions/total_predictions '
    corrects=np.where(y_true == y_predicted,1,0)
    corrects_num=np.sum(corrects)
    total_num=np.sum(np.ones(y_true.shape))
    output=corrects_num/total_num    
    return output

def precision(y_true,y_predicted):
    'precision = (FP)/(TP+FP) '
    TP=np.sum(np.where((y_predicted==1) & (y_true==1),1,0))
    FP=np.sum(np.where((y_predicted==1) & (y_true==-1),1,0))
    output=TP/(TP+FP)
    return output

def tpr(y_true,y_predicted):
    ' true positive rate = (TP)/(TP+FN) (aka recall)'
    TP=np.sum(np.where((y_predicted==1) & (y_true==1),1,0))
    FN=np.sum(np.where((y_predicted==-1) & (y_true==1),1,0))
    output=TP/(TP+FN)
    return output 

def fpr(y_true,y_predicted):
    ' false positive rate = (FP)/(FP+TN) '
    FP=np.sum(np.where((y_predicted==1) & (y_true==-1),1,0))
    TN=np.sum(np.where((y_predicted==-1) & (y_true==-1),1,0))
    output=FP/(FP+TN)
    return output

def f1(y_true,y_predicted):
    ' f1 = 2* (precision * recall) / (precision + recall) '
    prec=precision(y_true,y_predicted)
    rec=tpr(y_true,y_predicted)
    output=2*(prec*rec)/(prec+rec)
    return output 

def metrics_summary(y_true,y_predicted):
    acc=accuracy(y_true,y_predicted)
    prec=precision(y_true,y_predicted)
    tprate=tpr(y_true,y_predicted)
    fprate=fpr(y_true,y_predicted)
    f1score=f1(y_true,y_predicted)
    return acc,prec,tprate,fprate,f1score
