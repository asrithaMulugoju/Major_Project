from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score

def PM(ACC, NMI, F1, REC, PREC, gnd, res):
    CM = confusion_matrix(gnd, res)
    ACC.append(accuracy_score(gnd, res))
    NMI.append(normalized_mutual_info_score(gnd, res))
    F1.append(f1_score(gnd, res))
    REC.append(recall_score(gnd, res))
    PREC.append(precision_score(gnd, res))
    return ACC, NMI, F1, REC, PREC, CM