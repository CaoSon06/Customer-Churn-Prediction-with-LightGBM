# Customer Churn Prediction with LightGBM

Compare Logistic Regression and LightGBM, including hyperparameter tuning with RandomizedSearchCV and fine-tuning with early stopping, to predict bank customer churn (`Exited`). The notebook includes exploratory data analysis (EDA), model evaluation, F1-based threshold optimization, and SHAP explanations.

## Repository structure

```text
lightgbm-churn/
├── ML_lightGBM.ipynb   # Complete workflow; outputs cleared for readable diffs
├── requirements.txt
├── .gitignore
└── data/
    └── README.md      # Dataset setup instructions
```

## Run locally

Use Python 3.12 or 3.13. Run these commands from the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Place your dataset at `data/Churn_Modelling.csv`, open `ML_lightGBM.ipynb`, select the `.venv` kernel, and choose **Restart Kernel and Run All Cells**. If you need to register the kernel:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name lightgbm-churn --display-name "Python (lightgbm-churn)"
```

The first code cell configures paths, the number of search iterations (`N_ITER=40`), cross-validation folds (`CV=5`), and SHAP (`RUN_SHAP=True`). Set `N_ITER=2`, `CV=2`, and `RUN_SHAP=False` for a quick check. During the search, only one level uses all CPU cores to avoid nested parallelism.

The notebook creates an `artifacts/` directory containing the fine-tuned LightGBM pipeline, its decision threshold, and a model comparison CSV. The pipeline includes preprocessing; always use the matching threshold when converting probabilities to labels. Model saving, loading, and sample prediction examples appear at the end of the notebook.

## Methodology and limitations

- Use an 80/20 train/test split, stratified by `Exited`, with random seed 42.
- One-hot encode Geography/Gender and standardize numerical features for Logistic Regression.
- Tune LightGBM parameters using ROC-AUC and select the decision threshold using F1 on a validation split drawn from the training set.
- Use early stopping to determine the number of trees, then retrain on the full training set.
- Use the test ranking to report experimental results. The export uses the fine-tuned model and validation-derived threshold specified in the notebook; it does not automatically select a model from the test ranking.

The experimental method follows the original notebook: the validation split is drawn from the training set after hyperparameters have been tuned on that training set, and it is reused for early stopping and threshold selection. Validation scores therefore do not provide an independent estimate of performance. For stricter evaluation, hold out a separate validation set before tuning or use nested cross-validation, and reserve the test set for final evaluation.

The original notebook contained saved outputs; this repository clears them to reduce file size. The workflow has not been rerun on `Churn_Modelling.csv` because the dataset is absent from the workspace, so historical scores should not be treated as reproduced results. Dependency versions have major-version upper bounds; they are not a lockfile that exactly reproduces the original environment.

## Push to GitHub

Git has already been initialized in this directory. Create an empty repository on GitHub, then run:

```powershell
git add .
git commit -m "Add customer churn LightGBM notebook"
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

CSV datasets, model files, virtual environments, and the `artifacts/` directory are excluded by `.gitignore`. Before subsequent commits, choose **Clear Outputs of All Cells** and save the notebook to keep the repository small.
