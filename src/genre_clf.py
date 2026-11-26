"""Clasificación multi-etiqueta de géneros: datos, protocolo de evaluación y modelos clásicos (NB, LR).

Uso:  python src/genre_clf.py        # entrena NB y LR, escribe results/metrics.json y outputs/
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
THRESHOLDS = np.arange(0.10, 0.75, 0.025)


def load_data(data_dir=ROOT / "data"):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def clean(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def make_splits(train, seed=SEED):
    """60/20/20 train/val/holdout. Umbrales se eligen en val; las métricas se reportan en holdout."""
    texts = train["description"].fillna("").map(clean).to_numpy()
    labels = train["genre"].fillna("").map(lambda s: [g.strip() for g in s.split(",") if g.strip()])
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)
    X_tr, X_rest, Y_tr, Y_rest = train_test_split(texts, Y, test_size=0.4, random_state=seed)
    X_val, X_ho, Y_val, Y_ho = train_test_split(X_rest, Y_rest, test_size=0.5, random_state=seed)
    return dict(X_tr=X_tr, Y_tr=Y_tr, X_val=X_val, Y_val=Y_val, X_ho=X_ho, Y_ho=Y_ho, classes=mlb.classes_, mlb=mlb)


def apply_threshold(prob, t):
    """Umbral global; garantiza al menos un género por película."""
    pred = (prob >= t).astype(int)
    pred[np.arange(len(pred)), prob.argmax(1)] = 1
    return pred


def best_threshold(prob, Y):
    scores = [f1_score(Y, apply_threshold(prob, t), average="micro", zero_division=0) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def score(Y, pred, t):
    f1 = lambda avg: f1_score(Y, pred, average=avg, zero_division=0)
    return dict(
        micro_f1=f1("micro"), macro_f1=f1("macro"), samples_f1=f1("samples"),
        precision=precision_score(Y, pred, average="micro", zero_division=0),
        recall=recall_score(Y, pred, average="micro", zero_division=0),
        hamming_loss=hamming_loss(Y, pred), threshold=t,
    )


def evaluate(prob_val, prob_ho, s):
    t = best_threshold(prob_val, s["Y_val"])
    return score(s["Y_ho"], apply_threshold(prob_ho, t), t)


def fit_nb(s):
    vec = CountVectorizer(max_features=5000, stop_words="english")
    clf = OneVsRestClassifier(MultinomialNB(alpha=0.1)).fit(vec.fit_transform(s["X_tr"]), s["Y_tr"])
    return lambda X: clf.predict_proba(vec.transform(X))


def fit_lr(s):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english", min_df=2, sublinear_tf=True)
    clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)).fit(
        vec.fit_transform(s["X_tr"]), s["Y_tr"]
    )
    return lambda X: clf.predict_proba(vec.transform(X))


def baseline_most_frequent(s):
    """Referencia: predice siempre el género más frecuente del train."""
    top = s["Y_tr"].mean(0).argmax()
    pred = np.zeros_like(s["Y_ho"])
    pred[:, top] = 1
    return score(s["Y_ho"], pred, float("nan"))


def run(train, seed=SEED):
    s = make_splits(train, seed)
    results = {"Más frecuente (Drama)": baseline_most_frequent(s)}
    predictors = {}
    for name, fit in [("Naive Bayes (BoW)", fit_nb), ("Regresión logística (TF-IDF)", fit_lr)]:
        predict = fit(s)
        results[name] = evaluate(predict(s["X_val"]), predict(s["X_ho"]), s)
        predictors[name] = predict
    return s, results, predictors


def save_metrics(results, path=ROOT / "results" / "metrics.json"):
    path.parent.mkdir(exist_ok=True)
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **results}, indent=2, ensure_ascii=False))


def to_markdown(results):
    cols = ["micro_f1", "macro_f1", "samples_f1", "precision", "recall", "hamming_loss", "threshold"]
    rows = ["| Modelo | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, m in results.items():
        rows.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    train, test = load_data()
    s, results, predictors = run(train)
    save_metrics(results)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    X_test = test["description"].fillna("").map(clean).to_numpy()
    for name, key in [("Naive Bayes (BoW)", "nb"), ("Regresión logística (TF-IDF)", "lr")]:
        pred = apply_threshold(predictors[name](X_test), results[name]["threshold"])
        genres = [", ".join(g) for g in s["mlb"].inverse_transform(pred)]
        pd.DataFrame({"movie_name": test["movie_name"], "genre": genres}).to_csv(out / f"test_preds_{key}.csv", index=False)
    sys.stdout.reconfigure(encoding="utf-8")
    print(to_markdown(results))
