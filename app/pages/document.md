# IMDB Sentiment Classification with BGE Embeddings + MLflow Tracking

A sentiment analysis pipeline that converts IMDB movie reviews into dense
vector embeddings using `BAAI/bge-small-en-v1.5`, then trains traditional
machine learning classifiers (Logistic Regression, K-Nearest Neighbors,
Random Forest) on those embeddings to predict positive vs. negative
sentiment. All experiments — parameters, metrics, and model artifacts — are
tracked with **MLflow**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Embedding Generation](#embedding-generation)
- [Models Trained](#models-trained)
- [MLflow Tracking](#mlflow-tracking)


---

## Overview

This project answers a simple question: *can a small, frozen sentence
embedding model plus a classical ML classifier match the convenience of
fine-tuning a full transformer for sentiment classification?*

Instead of fine-tuning a large language model end-to-end, the pipeline:

1. Encodes each IMDB review into a fixed-size dense vector using the
   pretrained `BAAI/bge-small-en-v1.5` embedding model (no training of the
   embedding model itself — it is used purely as a frozen feature extractor).
2. Feeds those vectors into traditional ML classifiers (Logistic Regression,
   KNN, Random Forest) to learn the positive/negative sentiment boundary.
3. Logs every training run — hyperparameters, evaluation metrics, and the
   serialized model artifact — to MLflow for comparison and reproducibility.

This decouples representation learning (handled by the pretrained embedding
model) from classification (handled by fast, interpretable, easily-tuned
sklearn models), which keeps training cheap and the experiment loop fast.

---

## Architecture

```
                 ┌─────────────────────┐
 IMDB reviews →  │ BAAI/bge-small-en-   │ →  384-dim embedding vectors
 (raw text)      │ v1.5 (frozen encoder)│      (L2-normalized)
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Traditional ML      │
                 │  classifier          │
                 │  (LogReg / KNN / RF)│
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   MLflow Tracking    │
                 │  params · metrics ·  │
                 │  model artifacts     │
                 └─────────────────────┘
                            │
                            ▼
                 Positive / Negative label
```

The embedding step and the classifier-training step are split across two
notebooks (see [Project Structure](#project-structure)), so embeddings are
computed once and reused across multiple model experiments instead of being
recomputed for every training run.

---

## Dataset

- **Source:** [IMDB Movie Reviews dataset](https://huggingface.co/datasets/imdb), loaded via
  Hugging Face `datasets.load_dataset('imdb')`.
- **Task:** Binary sentiment classification — `0` = negative, `1` = positive.
- **Splits used:** `train` and `test`, as provided by the dataset (25,000
  reviews each).
- Rows are shuffled with `df.sample(frac=1, random_state=42)` for both the
  train and test embedding frames before training, to remove any ordering
  bias and keep results reproducible.

---

## Embedding Generation

Embeddings are generated in `embeddings-with-ml.ipynb` using
**LangChain's `HuggingFaceEmbeddings`** wrapper around the BGE model.

```python
from langchain_huggingface import HuggingFaceEmbeddings

model = "BAAI/bge-small-en-v1.5"

embedding = HuggingFaceEmbeddings(
    model_name=model,
    model_kwargs={"device": "cuda"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": BATCH_SIZE   # 64
    }
)
```

**Key details:**

| Setting | Value |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Output dimension | 384 |
| Device | CUDA (GPU) |
| Batch size | 64 |
| Normalization | L2-normalized embeddings (`normalize_embeddings=True`) |

Each review's text is embedded in batches (with a `tqdm` progress bar), and
the resulting vectors are stacked into a NumPy array of shape `(N, 384)`.
This is converted into a DataFrame where each of the 384 columns is one
embedding dimension, plus a final `label` column carrying the sentiment
target:

```python
def to_df(data, portion: str = "train"):
    data = data[portion]
    text = data["text"]
    label = data["label"]

    all_embeddings = []
    for i in tqdm(range(0, len(text), BATCH_SIZE), desc=f"Embedding [{portion}]"):
        batch = text[i:i + BATCH_SIZE]
        embeds = embedding.embed_documents(batch)
        all_embeddings.extend(embeds)

    emb_array = np.array(all_embeddings)        # (N, 384)
    dim = emb_array.shape[1]

    df = pd.DataFrame(emb_array, columns=list(range(dim)))
    df["label"] = label
    df["text"] = text
    return df
```

The resulting embedding tables are persisted to disk so they can be reused
without re-embedding:

- `data.csv` — embedded training set (384 feature columns + `label`)
- `data_test.csv` — embedded test set (384 feature columns + `label`)

These two CSVs are the handoff point between the embedding notebook and the
model-training notebook.

---

## Models Trained

Three traditional ML classifiers are trained on the BGE embedding vectors,
each in its own MLflow experiment, each swept across a hyperparameter grid:

### 1. Logistic Regression
- **MLflow experiment name:** `Logistic Regression`
- **Hyperparameter grid:**

  | Param | Values |
  |---|---|
  | `C` | `0.001, 0.01, 0.1, 1, 10, 100` |
  | `penalty` | `l1, l2` |
  | `solver` | `liblinear` |

  `max_iter=1000` is fixed for all runs.
- **Run naming convention:** `LR_{C}_{penalty}_{solver}`

### 2. K-Nearest Neighbors
- **MLflow experiment name:** `K neighbours`
- **Hyperparameter grid:** `n_neighbors` swept from `1` to `14`
- **Run naming convention:** `KNN_{n_neighbors}`

### 3. Random Forest
- **MLflow experiment name:** `Random Forest`
- **Hyperparameter grid:**

  | Param | Values |
  |---|---|
  | `n_estimators` | `100, 200, 500` |
  | `max_depth` | `None, 10, 20, 50` |
  | `min_samples_split` | `2, 5, 10` |
  | `min_samples_leaf` | `1, 2, 4` |
  | `max_features` | `sqrt, log2` |

  This is a full grid sweep (3 × 4 × 3 × 3 × 2 = 216 combinations), each
  logged as a separate MLflow run.

All three classifiers use scikit-learn implementations
(`LogisticRegression`, `KNeighborsClassifier`, `RandomForestClassifier`) and
are trained directly on the 384-dimensional embedding features (`X`) against
the binary sentiment label (`y`).

---

## MLflow Tracking

Every training run logs three categories of information to MLflow:

### Parameters (`mlflow.log_param`)
The specific hyperparameter combination used for that run (e.g. `C`,
`penalty`, `solver` for Logistic Regression; `neighbours` for KNN;
`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
`max_features` for Random Forest).

### Metrics (`mlflow.log_metric`)
- **`Accuracy`** — `accuracy_score(y_test, y_pred)` is logged for every run
  across all three models, computed against the held-out embedded test set
  (`data_test.csv`).

> **Note:** The Logistic Regression sweep in `embeddings-with-ml.ipynb`
> originally attempted to log regression-style metrics (`mae`, `mse`,
> `rmse`, `mpe`, `R2_score`) which are not meaningful for a classification
> task. The consolidated pipeline in `mlflow.ipynb` corrects this and
> standardizes on **`Accuracy`** as the tracked metric for all three
> classifiers, which is the recommended metric to rely on.

### Artifacts (`mlflow.sklearn.log_model`)
The trained sklearn model object is logged as an MLflow model artifact for
every run, along with an `input_example` (a single row of `X`) so the
model's expected input schema is captured automatically:

```python
mlflow.sklearn.log_model(model, name="model", input_example=X[:1])
```

This makes every run's model independently loadable and servable later via
`mlflow.sklearn.load_model(...)` or `mlflow pyfunc serve`, without needing
to re-run training.

### Experiment Organization
Each model family is tracked under its own **named MLflow experiment**, so
runs can be compared within a model type via the MLflow UI:

| MLflow Experiment | Model |
|---|---|
| `Logistic Regression` | `LogisticRegression` |
| `K neighbours` | `KNeighborsClassifier` |
| `Random Forest` | `RandomForestClassifier` |

To launch the MLflow UI and compare runs:

```bash
mlflow ui
```

Then open `http://localhost:5000` in a browser to view runs, sort/filter by
metric, and download/inspect logged model artifacts.

---