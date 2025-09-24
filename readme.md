## Requirements

Install dependencies using:

``` bash

pip install -r requirements.txt
```


## How to Run

``` bash

python main.py
```

This will:
1. Load Folds 1-4 for training, fold 5 for testing
2. Preprocess text (unigrams + bigrams)
3. Train and evaluate all models
4. Save results to
	- `results_summary`(holdout results) 
	- `cv_results_summary` (cross-validation results)
5. Save confision matrices to `CM/` folder

## Outputs

- **CSV Summaries** of performance (accuracy, precision, recall, F1, best, hyperparameters)
- **Confusion matrices** saves as `.png`
- **Console logs** of metrics and per-class performance

## Reproducibility
- Running `main.py` will reproduce all experiments
- Hyperparameter tuning and CV are included in each model
- Results may vary slightly due to random seeds ( fixed where possible)