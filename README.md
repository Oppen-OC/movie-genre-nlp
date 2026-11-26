# Clasificación de Géneros de Películas con NLP

Proyecto final de Procesamiento de Lenguaje Natural. Clasifica géneros de películas a partir de sus descripciones en texto, utilizando tres enfoques distintos: Naive Bayes, Regresión Logística y un modelo Transformer (RoBERTa).

## Descripción del problema

Dado el nombre y la descripción de una película, el modelo predice uno o más géneros cinematográficos (clasificación multi-etiqueta). El dataset contiene 8,475 películas de entrenamiento con 18 géneros posibles.

**Géneros disponibles:** Action, Adventure, Animation, Comedy, Crime, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western

## Estructura del repositorio

```
ProyectoFinal/
└── nlp_movie_genre/
    ├── data/
    │   ├── train.csv              # 8,475 películas con géneros y descripciones
    │   └── test.csv               # 942 películas sin etiqueta (para predicción)
    └── src/
        ├── main1.ipynb            # Naive Bayes + Regresión Logística (baseline)
        ├── main2.ipynb            # Regresión Logística mejorada
        ├── main - copia.ipynb     # Los 3 modelos: NB, LR y Transformer
        ├── dataset_test_preds.csv      # Predicciones del modelo LR mejorado
        ├── dataset_test_preds_lr.csv   # Predicciones de Regresión Logística
        └── dataset_test_preds_nb.csv   # Predicciones de Naive Bayes
```

## Modelos implementados

### Modelo 1 — Naive Bayes con CountVectorizer

- Vectorización con bolsa de palabras (5,000 features)
- Clasificador `ComplementNB` envuelto en `OneVsRestClassifier`
- Predicción por umbral de probabilidad

| Conjunto | Precisión | Recall |
|----------|-----------|--------|
| Entrenamiento | 70% | 81% |
| Validación | 57% | 64% |

### Modelo 2 — Regresión Logística con TF-IDF

- TF-IDF con 5,000 features
- `LogisticRegression` con `OneVsRestClassifier`
- Se evalúan múltiples umbrales (0.2, 0.25, 0.3)

| Umbral | Precisión (val) | Recall (val) |
|--------|-----------------|--------------|
| 0.2 | ~40% | ~86% |
| 0.3 | ~60% | ~58% |

### Modelo 3 — Regresión Logística mejorada (`main2.ipynb`)

Mejoras sobre el modelo anterior:
- TF-IDF con **8,000 features** y **bigramas** (`ngram_range=(1,2)`)
- `min_df=2` para filtrar términos poco frecuentes
- `max_iter=2000`, `C=2`, `class_weight='balanced'`

### Modelo 4 — Transformer (RoBERTa)

- Modelo base: `roberta-base` (Hugging Face)
- Fine-tuning con clasificación multi-etiqueta (activación sigmoid)
- Entrenado sobre una muestra de 2,000 películas por limitaciones computacionales
- 1 época de entrenamiento (~3,000 segundos)
- Umbral de predicción: 0.3

## Datos

| Archivo | Filas | Columnas |
|---------|-------|----------|
| `train.csv` | 8,475 | `movie_name`, `genre`, `description` |
| `test.csv` | 942 | `movie_name`, `description` |

Los géneros en `train.csv` están en formato de cadena separada por comas, que se convierte a formato multi-etiqueta con `MultiLabelBinarizer`.

## Dependencias

```
pandas
scikit-learn
torch
transformers
datasets
```

No hay `requirements.txt` incluido. Se recomienda instalar con:

```bash
pip install pandas scikit-learn torch transformers datasets
```

## Uso

Abrir los notebooks en orden:

1. **`main1.ipynb`** — Exploración inicial y modelos baseline
2. **`main2.ipynb`** — Regresión Logística optimizada con predicciones sobre el test set
3. **`main - copia.ipynb`** — Comparación completa de los tres enfoques incluyendo el Transformer

Los archivos de predicción generados (`dataset_test_preds_*.csv`) contienen, para cada película del test set, los géneros predichos por cada modelo.

## Resultados

El modelo de Regresión Logística con TF-IDF mejorado ofrece el mejor equilibrio entre precisión y recall para la mayoría de umbrales. El Transformer (RoBERTa) fue entrenado con datos reducidos, por lo que sus resultados son preliminares y mejorables con más recursos computacionales o más épocas de entrenamiento.
