from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score
from helper import save_pred_and_labels
from scipy.special import expit
import constants


def compute_metrics_ACD(results):

    def compute_metrics(eval_pred):
        predictions, lab = eval_pred
        save_pred_and_labels(predictions, lab, results)
        predictions = (expit(predictions) > 0.5)
        labels = [l == 1 for l in lab]
        accuracy = accuracy_score(labels, predictions)

        f1_macro = f1_score(labels, predictions,
                            average="macro", zero_division=0)
        f1_micro = f1_score(labels, predictions,
                            average="micro", zero_division=0)
        f1_weighted = f1_score(labels, predictions,
                               average="weighted", zero_division=0)

        hamming = hamming_loss(labels, predictions)

        metrics = {
            "hamming_loss": hamming,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
        }

        return metrics

    return compute_metrics
