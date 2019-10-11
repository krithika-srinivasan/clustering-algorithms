from sklearn.metrics import adjusted_rand_score, jaccard_score

def rand_score(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)

def jaccard_coeff(labels_true, labels_pred):
    # Average is set to "weighted" here to take into label imbalance
    return jaccard_score(labels_true, labels_pred, average="weighted")

