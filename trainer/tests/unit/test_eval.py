import numpy as np

from trainer.train import get_metrics


def test_metrics_computed_correctly():
    preds = [0.99, 0.98, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    m = get_metrics(preds, labels)

    assert m["AUROC"] == 1.0, (
        "Failure in computing correct AUROC from preds and labels."
    )
    assert m["AUPRC"] == 1.0, (
        "Failure in computing correct AUPRC from preds and labels."
    )
    assert m["sens_at_95_spec"] == np.float64(1.0), (
        "Failure in computing correct sensibility at specificity 95% from preds and labels."
    )
