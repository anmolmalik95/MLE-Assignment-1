# MLE Assignment 1: Feature and Label Store Data Pipeline

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.5.5-E25A1C?logo=apachespark&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![JupyterLab](https://img.shields.io/badge/JupyterLab-4.3.5-F37626?logo=jupyter&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Medallion-C7A33B)

CS611 Machine Learning Engineering, Assignment 1. A production-style data pipeline that turns four raw source files into a machine-learning-ready **feature store** and reuses the Lab 2 **label store**, built with PySpark and Docker on the Medallion Architecture, for predicting loan default at the point of application.

> Grader note: the single-line repository link required by the assignment is in `Readme.txt`. This `README.md` is the project overview and does not replace it.

## Overview

The bank lends cash loans and needs to predict, at application time, whether a customer will default. The outcome is only known months later, so the pipeline is built to keep those two moments apart and avoid data leakage. This repository covers the data preparation only; the model itself is the next assignment.

## Architecture

The pipeline follows the Medallion Architecture across three layers:

- **Bronze**: ingest only, no transformation. Each raw source is landed exactly as received and partitioned by month, so any value can be traced back to the original.
- **Silver**: cleaned and standardised. Corrupt values are stripped and cast, impossible values are nulled rather than guessed, and missingness is kept as a signal instead of dropping rows.
- **Gold**: two model-ready tables, each one row per loan.
  - `feature_store` (8,974 x 47): each loan's features are joined on as of its own application date, so nothing dated after the decision can enter the row.
  - `label_store` (8,974 x 5): 30 or more days past due at month-on-book 6, reused from Lab 2 unchanged.

The two gold tables align one to one on `loan_id`. A simple classification model trained on them reaches a test AUC of about 0.886 with no leakage signature, which is the sanity check that the feature store is machine-learning compatible.

## Repository structure

```
.
├── main.py                 # runs the whole pipeline: bronze -> silver -> gold
├── utils/
│   ├── data_processing_bronze_table.py
│   ├── data_processing_silver_table.py
│   └── data_processing_gold_table.py
├── data/                   # the four raw source CSVs
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── Readme.txt              # one-line repo link (assignment requirement)
└── .gitignore              # datamart/ is generated at runtime, not committed
```

## How to run

```
docker-compose build
docker-compose up
```

`docker-compose up` prints a JupyterLab link. Open it, then in a terminal inside JupyterLab run:

```
python main.py
```

This builds a `datamart/` folder containing `bronze/`, `silver/` and `gold/` subfolders. Reruns overwrite cleanly and the transforms are deterministic, so the same inputs always produce the same datamart.

## Data leakage

Three leakage types are handled by design:

- **Target leakage**: no feature is built from the loan outcome or any post-application loan record.
- **Train-test contamination**: no scaling, encoding or splitting happens in this pipeline; that is deferred to the model pipeline.
- **Temporal leakage**: the as-of rule attaches a customer feature only if its snapshot date is on or before the loan's application date, and the label is observed six months later.

## Pipeline output

| Table | Shape | Grain | Notes |
| --- | --- | --- | --- |
| `gold/feature_store` | 8,974 x 47 | one row per loan | features as of the application date |
| `gold/label_store` | 8,974 x 5 | one row per loan | 30dpd at month-on-book 6, reused from Lab 2 |

The stores join one to one on `loan_id` to form the training table consumed by the downstream model pipeline.

## Author

Anmol Malik. CS611 Machine Learning Engineering, Assignment 1.
