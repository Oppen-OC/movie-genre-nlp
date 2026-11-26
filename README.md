# Movie Genre NLP

Clasificación multi-etiqueta de géneros de películas (18 géneros) a partir de su sinopsis, comparando Naive Bayes, regresión logística y un Transformer (RoBERTa) bajo el mismo protocolo de evaluación.

Proyecto del curso *Procesamiento del Lenguaje Natural y Text Mining* (UPV), basado en la competición CodaLab *NLP & Text-Mining: Movie Genre Identification Task*. Se hizo en pareja; este repo es la versión consolidada y reproducible.

## Enfoque

```
description ──► limpieza (minúsculas, solo letras)
   │
   ├─► CountVectorizer (5k)          ─► OvR MultinomialNB
   ├─► TF-IDF (20k, 1-2 gramas)      ─► OvR LogisticRegression
   └─► tokenizer roberta-base (128)  ─► fine-tuning con BCE (sigmoide por género)
                                         │
        probabilidades ─► umbral global (elegido en val) ─► ≥1 género por película
```

**Protocolo.** `train.csv` (8.475 películas) se divide 60/20/20 en train/val/holdout con semilla 42. El umbral global se elige maximizando micro-F1 en *val*; las métricas se reportan en *holdout*, que ningún modelo ha visto para decidir nada. El `test.csv` (942 películas) no tiene etiquetas: solo se usa para generar predicciones (`outputs/`).

## Resultados (holdout, 1.695 películas)

Salen de `results/metrics.json`, generado por `src/genre_clf.py` y `src/roberta.py`.

| Modelo | micro F1 | macro F1 | samples F1 | precisión | recall | Hamming loss | umbral |
|---|---|---|---|---|---|---|---|
| Más frecuente (Drama) | 0.254 | 0.035 | 0.280 | 0.465 | 0.175 | 0.152 | – |
| Naive Bayes (BoW) | 0.591 | 0.501 | 0.579 | 0.550 | 0.638 | 0.131 | 0.250 |
| Regresión logística (TF-IDF) | 0.580 | 0.408 | 0.574 | 0.504 | 0.683 | 0.146 | 0.225 |
| RoBERTa (5.085 train, 3 épocas) | 0.650 | 0.537 | 0.644 | 0.608 | 0.699 | 0.111 | 0.325 |

## Arranque rápido

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/genre_clf.py        # NB + LR → results/metrics.json, outputs/test_preds_*.csv (~segundos, CPU)
pytest -q                      # comprobaciones (también en CI)

# opcional, GPU recomendada
pip install -r requirements-roberta.txt
python src/roberta.py --epochs 3
```

`notebooks/movie_genre.ipynb` recorre la exploración y los resultados.

## Estructura

```
data/            train.csv, test.csv (competición CodaLab)
src/genre_clf.py datos, splits, umbral, métricas, NB y LR
src/roberta.py   fine-tuning de roberta-base sobre los mismos splits
notebooks/       notebook de referencia (exploración + resultados)
results/         metrics.json con las métricas de la última ejecución
tests/           umbral, splits reproducibles y modelos > baseline
docs/            memoria del trabajo (memoria.tex)
```

## Decisiones de diseño

- **Un solo split con tres partes** para no tunear el umbral y reportar sobre los mismos datos.
- **Umbral global**, no por género: simple y suficiente; los géneros minoritarios (TV Movie, Western) salen perjudicados.
- **Mínimo un género por película** (el más probable), porque todas las películas tienen al menos uno.
- **Baseline "Drama"** como referencia: micro-F1 de 0.25 muestra cuánto aporta cada modelo.

## Límites conocidos

- Una única semilla y un único split: las diferencias de pocos puntos (NB vs LR) no son concluyentes.
- Sin lematización ni spaCy (la memoria usó spaCy en una iteración); aquí la limpieza es solo regex para mantener el pipeline sin descargas.
- RoBERTa se entrena con 128 tokens máximo y sin búsqueda de hiperparámetros. Difiere de la memoria original (2.000 muestras, 1 época, sin validación).
- Las cifras de la memoria (`docs/memoria.tex`) vienen de envíos a CodaLab y de otro protocolo; no son comparables con esta tabla.

## Licencia

MIT. Los datos pertenecen a la competición de origen.
