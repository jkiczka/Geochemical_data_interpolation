# Interpolation of Geochemical data

This project focuses on improving interpolation and spatial prediction of geochemical variables (e.g., iodide or methane concentrations in oceans) by replacing conventional random train-test splits with **Patch-based Cross Validation (Patch-CV)**. Instead of predicting point-wise values, the goal is to predict **entire spatial patches**, better reflecting real-world generalization in unseen regions.

We experiment with a range of models including:
- **Encoder + MLP** (baseline)
- **Masked Autoencoders (MAE)**
- **CNN-based spatial models**
- **Graph Attention Networks (GAT)**
- **TabPFN** (for strong tabular baselines)

---

## Project Inspiration

This project was inspired by the following key publications:

- **Yang et al., 2020** – *A machine-learning-based global sea-surface iodide distribution* (<- this one mostly)
- **Weber et al., 2019** – *Global ocean methane emissions dominated by shallow coastal waters*
- **Sherwen et al., 2019** – *A machine-learning-based global sea-surface iodide distribution*

Yang's work highlighted the importance of global and regional prediction of oceanic compounds using machine learning, but mostly relied on random sampling which can overestimate model performance. We aim to improve this with **Patch-CV** evaluation and better model generalization.

---

## Project Structure
```
root/
│
├── data/                                                      # Geospatial data inputs
│ ├── processed/                                               # (Usually normalized and joined)
│ └── raw/                                                     # Original untouched datasets
│
├── models/                                                    # Saved trained models (non here)
│
├── notebooks/
│ └── WOA+YANG/                                                # Main exploratory and experimental notebooks
│ ├── baseline.ipynb
│ ├── mae_predict(joined_training).ipynb
│ └── ...                                                       # Many others notebooks
│
├── results/                                                    # Output predictions, metrics, visualizations
│
├── src/                                                        # Core code logic
│ ├── data_pipeline.py                                          # Preprocessing, feature engineering, patch extraction
│ ├── main.py
│ ├── main_grid.py                                              # Grid creation over datasets
│ ├── main_join.py                                              # Dataset merging/joining logic
│ ├── utils.py                                                  # Common helpers
│ └── init.py
│
├── .gitignore
├── README.md
```

---

## Key Features

- ✅ Patch-based Cross Validation: more realistic spatial split.
- ✅ Multiple model families for comparison.
- ✅ Jupyter Notebooks for rapid experimentation.
- ✅ Modular Python scripts in `src/` for production-grade execution.

---
