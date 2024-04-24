import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay

def roc_evaluation(targets, probs):
    # 1-label_score=probability of the label being a label issue (i.e. noisy label)
    pred_probs = [1-prob for prob in probs] if isinstance(probs, list) else 1-probs
    fpr, tpr, thresholds = roc_curve(targets, pred_probs)
    aucroc = auc(fpr, tpr)

    # Youden's J statistic to find the best threshold
    J = tpr-fpr
    J_max_idx = J.argmax()
    best_threshold = thresholds[J_max_idx]
    best_tpr = tpr[J_max_idx]
    best_fpr = fpr[J_max_idx]

    fig = plt.figure(1)
    plt.scatter(best_fpr, best_tpr, color='red', label=f"(fpr, tpr, thresh) = ({best_fpr:.2f}, {best_tpr:.2f}, {best_threshold:.2f})", marker="x")
    plt.plot((0, 1), (0, 1), 'k--', label="Random Guess (AUC = 0.5)")
    plt.plot(fpr, tpr, label=f"Classifier (AUC = {aucroc:.2f})")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    return aucroc, best_threshold, fig 

def classification_evaluation(targets, preds):
    """
    It returns the classification report from sklearn.metrics.classification_report.
    Also, it plots the examples of FP and FN of the CL, i.e. the examples that the CL failed to identify as the noisy labels.

    Args:
    - targets: np.array of shape (N, ), the ground truth labels
    - probs: np.array of shape (N, num_classes), the predicted probabilities

    Returns:
    - conf_mat_fig: matplotlib.figure.Figure, the figure containing the confusion matrix
    - cls_report: dict, the classification report from sklearn.metrics.classification_report
    """
    cls_report = classification_report(targets, preds, output_dict=True)
    conf_mat = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(conf_mat)
    conf_mat_fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.tight_layout()
    return conf_mat_fig, cls_report