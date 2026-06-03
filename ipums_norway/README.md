# Homonym resolution models. Submitted to SBBD 2026

This repository contains the data and modeling pipeline to identify and relate people (homonyms/kinship) using historical data from Norway (IPUMS). The solution builds candidate pairs, applies heuristic scores, and trains Machine Learning models for ranking (Learning to Rank).

Make sure you have the following files in the root of the `norway/` directory before running the codes:

* `ipumsi_00002.xml` and `ipumsi_00002.dat.gz` (Raw IPUMS databases).
* `municipios_noruega.csv` (Municipality mapping).
* Pre-computed embeddings files: `IpumsNORWAY_distinct_municipio_emb.pkl` and `IpumsNORWAY_distinct_profissao_emb.pkl`.

Main dependencies:

* pip install pandas numpy scikit-learn xgboost lightgbm shap ipumspy tqdm

In general, the code was developed in a Jupyter Lab environment, on a machine with 30GB of RAM (necessary for the initial data processing).
It is necessary to execute all cells, following the numerical order; some cells are for testing and verification purposes only.

## Execution Order and Code Description

The notebooks have been numbered to be executed sequentially.

### [`1.gera_municipios.ipynb`](https://www.google.com/search?q=1.gera_municipios.ipynb)

**Objective:** Extraction and formatting.
Reads the XML data containing the categories, codes, and geographic labels of the counties and municipalities of Norway for initial structuring.

### [`2.gera_pares.ipynb`](https://www.google.com/search?q=2.gera_pares.ipynb)

**Objective:** Data cleaning and crossing.
Imports the raw data, filters necessary columns, and creates unique IDs. The main methodological points discussed in the article are structured here: detailed location (where it is possible to check in which columns the person was born and currently lives), age, and occupation/kinship (often mapped by the `RELATED` column). The script handles missing values and generates the list of homonyms (candidates) for each head of household.

### [`3.explode.ipynb`](https://www.google.com/search?q=3.explode.ipynb)

**Objective:** Feature Creation and Splits.
Generates all combinations (crossings) between the heads of household and the candidate lists. Calculates similarity scores using the *embeddings* and estimates age differences. Performs data splitting through Cross-Validation (5 splits), using a proportion of 90% of the subsets for training and 10% distributed for testing and validation.

### [`4.valida_splits.ipynb`](https://www.google.com/search?q=4.valida_splits.ipynb)

**Objective:** Integrity check.
Ensures that the Train, Validation, and Test splits were generated correctly, checking the number of IDs, class balance, null counts, and ensuring there is no data leakage.

### [`5.experimento.ipynb`](https://www.google.com/search?q=5.experimento.ipynb) and [`copia_5.experimento.ipynb`](https://www.google.com/search?q=copia_5.experimento.ipynb)

**Objective:** Model Training and Inference.
Loads the generated splits. Trains *Learning to Rank* algorithms (such as LightGBM and XGBoost) and classification algorithms. Generates MRR (Mean Reciprocal Rank) and Recall evaluations on the test partitions.

### [`6.lgbm_explainer.ipynb`](https://www.google.com/search?q=6.lgbm_explainer.ipynb)

**Objective:** Model Interpretability.
Loads the trained LightGBM model and uses the **SHAP** library to extract the importance of each variable (feature) in the ranking decision process.

### [`7.estatisticas.ipynb`](https://www.google.com/search?q=7.estatisticas.ipynb)

**Objective:** Obtaining descriptive metrics.
Extracts percentiles, distribution, and quantity of input information, such as the average size of homonym lists, total number of people, and total analyzed pairs, preparing the results in tabular format.
