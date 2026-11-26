"""Fine-tuning de roberta-base multi-etiqueta sobre los mismos splits que NB/LR.

Uso:  python src/roberta.py [--epochs 3] [--n-train N]     (requiere requirements-roberta.txt; GPU recomendada)
"""
import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from genre_clf import SEED, evaluate, load_data, make_splits, save_metrics, to_markdown

MODEL = "roberta-base"


def predict_proba(model, tok, texts, device, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(list(texts[i:i + bs]), truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            out.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    return np.concatenate(out)


def main(epochs, n_train, lr=2e-5, bs=16):
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, _ = load_data()
    s = make_splits(train)
    X, Y = s["X_tr"][:n_train], s["Y_tr"][:n_train]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(s["classes"]), problem_type="multi_label_classification"
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(X))
        for i in range(0, len(X), bs):
            idx = order[i:i + bs]
            enc = tok(list(X[idx]), truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            loss = model(**enc, labels=torch.tensor(Y[idx], dtype=torch.float, device=device)).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(f"epoch {ep + 1}/{epochs} loss={loss.item():.4f}", flush=True)
    name = f"RoBERTa ({len(X)} train, {epochs} ep)"
    res = {name: evaluate(predict_proba(model, tok, s["X_val"], device), predict_proba(model, tok, s["X_ho"], device), s)}
    save_metrics(res)
    sys.stdout.reconfigure(encoding="utf-8")
    print(to_markdown(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=10**9)
    a = ap.parse_args()
    main(a.epochs, a.n_train)
