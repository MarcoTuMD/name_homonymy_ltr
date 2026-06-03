# Homonym Disambiguation and Entity Resolution Pipelines (SBBD 2026)

This repository contains the complete data engineering pipelines, feature extraction workflows, dataset generation scripts, and machine learning ranking models utilized in the paper submitted to the **41st Brazilian Symposium on Databases (SBBD 2026)**.

The project addresses the critical challenge of nominal ambiguity resolution (homonym disambiguation) and kinship topology reconstruction across multiple diverse, large-scale, and historical data sources. It includes deterministic heuristics, advanced feature engineering (such as semantic vector embeddings for locations and occupations), and supervised *Learning to Rank* framework evaluations (e.g., XGBoost, LightGBM).

---

## Repository Structure

The repository is structured into modular pipelines representing each distinct experimental setup described in the paper:

- `/ipums_iceland`: Pipeline scripts and notebooks for IPUMS Iceland (1729) data processing, candidate pair generation, and modeling.
- `/ipums_norway`: Sequential pipeline notebooks for IPUMS Norway (1900) data extraction, candidate pair generation, data splitting, and model execution.
- `/ipums_united_states`: Pipeline scripts and notebooks for IPUMS United States (1930) data processing, candidate pair generation, and modeling.
- `/wikidata`: Multi-phase data ingestion, semantic label mapping, embedding extraction, and statistical validation pipeline for Wikidata subsets.

*(Note: The Brazilian Civil Registry data was utilized in the study for comparative analysis, but its processing pipeline is not included in this public repository structure).*

---

## Dataset Characteristics & Relationship Distributions

To ensure consistency and comparative rigor across all experiments, relationship labels from all sources have been normalized into a standardized English taxonomy. Below is the relational distribution for each of the five datasets utilized in this study.

### 1. Brazilian Civil Registry

| Relationship Label | Count | Percentage (%) |
| :--- | :---: | :---: |
| Sibling | 130,942 | 46.88% |
| Uncle/Aunt | 44,040 | 15.77% |
| Mother | 40,108 | 14.36% |
| Cousin | 17,758 | 6.36% |
| Nephew/Niece | 17,684 | 6.33% |
| Child | 16,902 | 6.05% |
| Grandparent | 8,472 | 3.03% |
| Spouse | 1,496 | 0.54% |
| Grandchild | 1,042 | 0.37% |
| Father | 425 | 0.15% |
| Parent-in-law | 421 | 0.15% |
| **TOTAL** | **279,290** | **100.00%** |

### 2. IPUMS Iceland (1729)

| Relationship Label | Count | Percentage (%) |
| :--- | :---: | :---: |
| Child | 1,778 | 70.75% |
| Spouse/Partner | 601 | 23.92% |
| Other Relative | 134 | 5.33% |
| **TOTAL** | **2,513** | **100.00%** |

### 3. IPUMS Norway (1900)

| Relationship Label | Count | Percentage (%) |
| :--- | :---: | :---: |
| Child (Biological) | 543,817 | 69.37% |
| Spouse | 203,458 | 25.95% |
| Child (Adopted) | 9,451 | 1.21% |
| Parent | 6,639 | 0.85% |
| Grandchild | 6,617 | 0.84% |
| Parent-in-law | 5,097 | 0.65% |
| Sibling | 3,852 | 0.49% |
| Child-in-law | 2,036 | 0.26% |
| Grandparent | 998 | 0.13% |
| Stepchild | 770 | 0.10% |
| Nephew/Niece | 725 | 0.09% |
| Sibling-in-law | 304 | 0.04% |
| Stepparent | 118 | 0.02% |
| Cousin | 26 | < 0.01% |
| Stepsibling | 25 | < 0.01% |
| Great-grandchild | 13 | < 0.01% |
| Unmarried Partner | 2 | < 0.01% |
| **TOTAL** | **783,955** | **100.00%** |

### 4. IPUMS United States (1930)

| Relationship Label | Count | Percentage (%) |
| :--- | :---: | :---: |
| Child | 670,935 | 58.06% |
| Spouse | 330,766 | 28.62% |
| Grandchild | 28,761 | 2.49% |
| Sibling | 25,762 | 2.23% |
| Parent | 17,569 | 1.52% |
| Nephew/Niece | 15,200 | 1.32% |
| Sibling-in-law | 14,741 | 1.28% |
| Parent-in-law | 14,605 | 1.26% |
| Stepchild | 13,599 | 1.18% |
| Child-in-law | 12,524 | 1.08% |
| Cousin | 2,854 | 0.25% |
| Uncle/Aunt | 2,555 | 0.22% |
| Child (Adopted) | 2,282 | 0.20% |
| Other Relative | 1,086 | 0.09% |
| Grandparent | 696 | 0.06% |
| Grandnephew/Grandniece | 305 | 0.03% |
| Step-grandchild | 271 | 0.02% |
| Stepparent | 264 | 0.02% |
| Step/Half/Adopted Sibling | 259 | 0.02% |
| Nephew/Niece-in-law | 213 | 0.02% |
| Grandchild-in-law | 157 | 0.01% |
| Adopted (n.s.) | 134 | 0.01% |
| Step Child-in-law | 119 | 0.01% |
| **TOTAL** | **1,155,657** | **100.00%** |

### 5. Wikidata (English Subset)

| Relationship Label | Count | Percentage (%) |
| :--- | :---: | :---: |
| Child | 95,610 | 36.15% |
| Father | 51,586 | 19.50% |
| Sibling | 49,025 | 18.53% |
| Spouse | 39,437 | 14.91% |
| Mother | 20,770 | 7.85% |
| Other Relative | 8,079 | 3.05% |
| **TOTAL** | **264,507** | **100.00%** |

---

## Setup and Dependencies

The scripts are optimized for running in Python 3.10+ environments with high RAM availability (minimum 32GB recommended for initial data parsing and pairwise matrix explosions).
