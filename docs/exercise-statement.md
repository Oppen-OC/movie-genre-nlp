# Enunciado del trabajo final (resumen)

Trabajo Final de *Natural Language Processing & Text Mining* (MIABDA, ETSINF, UPV), curso 2026: **Movie Genre Identification Task**. Resumen con palabras propias del enunciado oficial, que se publica en el aula virtual del curso y no se redistribuye aquí.

## Tarea

Predecir el género (o géneros) de una película a partir de una breve sinopsis. Es una tarea **multi-clase y multi-etiqueta**, con 18 géneros posibles: Action, Adventure, Animation, Comedy, Crime, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War y Western.

Se parte de dos sistemas base, uno clásico y otro de deep learning, que los equipos deben intentar mejorar. Cada equipo entrega 3 resultados distintos, al menos uno basado en aprendizaje clásico. Los envíos (máximo 3) se hacen a una competición alojada en Codabench por la propia asignatura, en la que hay que registrarse para descargar los datos y subir predicciones.

## Datos

CSV con 9.417 películas y tres columnas: `movie_name`, `genre` (géneros separados por comas) y `description`. El profesorado reparte el conjunto en 90 % entrenamiento (8.475 filas, `data/train.csv`) y 10 % evaluación sin etiquetas (942 filas, `data/test.csv`). Las etiquetas del 10 % de evaluación solo las tiene la plataforma, así que aquí no se pueden calcular métricas sobre ese conjunto.

## Métricas oficiales

Accuracy (coincidencia exacta del conjunto de etiquetas), precision, recall y F1 con `average='macro'`, y Hamming loss. Se proporciona una función `compute_metrics` basada en scikit-learn. Este repo reporta además micro-F1 y samples-F1, que no son las métricas del ranking.

## Sistemas base

| Base | Enfoque | accuracy | precision | recall | F1 macro | Hamming loss |
|---|---|---|---|---|---|---|
| 1 | `MultinomialNB` + `CountVectorizer` (5.000 características, minúsculas, stop words inglesas) | 0,12 | 0,51 | 0,54 | 0,53 | 0,12 |
| 2 | Fine-tuning de `roberta-base` multi-etiqueta (5 épocas, lr 5e-5, batch 16) | 0,19 | 0,68 | 0,56 | 0,61 | 0,09 |

Son cifras sobre el conjunto de evaluación de la plataforma. **No son comparables** con la tabla de `README.md`, que usa un split interno del entrenamiento.

## Evaluación del trabajo

Se evalúa una **memoria** (6 puntos): 5 por su calidad (amplitud de la experimentación y originalidad) y 1 por los resultados obtenidos. Debe experimentarse con aprendizaje clásico y con Transformers. La memoria es un PDF de 5 a 8 páginas con título y autores, descripción del trabajo (herramientas, análisis exploratorio, preproceso, vectorización, algoritmos, hiperparámetros, tablas comparativas), valoración y conclusiones, bibliografía y enlace al código. Plazo: 8 de mayo (2026), 23:55.
