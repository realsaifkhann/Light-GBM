# LightGBM — Learning Notes

**Type:** Supervised Learning — Regression & Classification  
**Library:** lightgbm  
**Dataset:** California Housing (sklearn)

---

## What is LightGBM?

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework developed by Microsoft in 2017. It builds decision trees sequentially — each tree corrects the errors of the previous one. The final prediction is the combined output of all trees.

---

## Why LightGBM?

Regular gradient boosting was accurate but slow. LightGBM solved that with three improvements:

| Technique | Purpose |
|---|---|
| Histogram binning | Groups feature values into buckets for faster splitting |
| Leaf-wise growth | Expands only the most impactful leaf, not all leaves equally |
| GOSS sampling | Retains hard examples, drops easy ones to reduce computation |

---

## Important Hyperparameters

| Parameter | Role |
|---|---|
| `num_leaves` | Controls tree complexity — most important parameter |
| `n_estimators` | Number of trees to build |
| `learning_rate` | Contribution of each tree to the final answer |
| `max_depth` | Maximum depth a tree can grow |
| `min_child_samples` | Minimum rows required per leaf |
| `subsample` | Fraction of rows sampled per tree |
| `colsample_bytree` | Fraction of features sampled per tree |
| `reg_alpha` | L1 regularization — removes irrelevant features |
| `reg_lambda` | L2 regularization — smoothly shrinks leaf weights |

---

## LightGBM vs XGBoost

| | LightGBM | XGBoost |
|---|---|---|
| Tree growth | Leaf-wise | Level-wise |
| Speed | Faster | Slower |
| Memory | Lower | Higher |
| Small datasets | Can overfit | More stable |
| Categorical features | Native support | Manual encoding needed |

---

## Results on California Housing

| Metric | Score |
|---|---|
| MAE | 0.31 |
| R2 Score | 0.84 |

The model explained 84% of the variance in median house prices using default hyperparameters.

---

## Key Takeaways

- `num_leaves` is the single most critical hyperparameter in LightGBM
- Always set `n_estimators` high and rely on early stopping rather than guessing
- LightGBM handles missing values natively — no imputation needed before training
- Use `LGBMRegressor` for continuous targets and `LGBMClassifier` for categorical ones

