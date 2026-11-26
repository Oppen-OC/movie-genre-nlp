import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import genre_clf as g


def test_apply_threshold_always_one_label():
    prob = np.array([[0.1, 0.2, 0.05], [0.9, 0.8, 0.1]])
    pred = g.apply_threshold(prob, 0.5)
    assert pred.tolist() == [[0, 1, 0], [1, 1, 0]]


def test_splits_are_disjoint_and_reproducible():
    train, _ = g.load_data()
    a, b = g.make_splits(train), g.make_splits(train)
    assert len(a["X_tr"]) + len(a["X_val"]) + len(a["X_ho"]) == len(train)
    assert len(a["classes"]) == 18
    assert (a["Y_ho"] == b["Y_ho"]).all()


def test_pipeline_beats_baseline_on_sample():
    train, _ = g.load_data()
    _, res, _ = g.run(train.sample(2000, random_state=0))
    base = res["Más frecuente (Drama)"]["micro_f1"]
    for name in ("Naive Bayes (BoW)", "Regresión logística (TF-IDF)"):
        assert res[name]["micro_f1"] > base + 0.1
