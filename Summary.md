# Practical Machine Learning Final Project — Summary

## Pipeline

The end-to-end pipeline now runs in `main.py` as:

1. **Feature engineering** — build the modeling table from `health_and_wellness_no_outliers.csv`. Product `Name` is now represented by both the old text statistics (`name_length`, `name_word_count`) and 16 PCA-compressed sentence embeddings (`name_semantic_pca_*`) from `paraphrase-multilingual-MiniLM-L12-v2`.
2. **Split** — stratified train/test split on `log_price_thb` (80/20, seed 42).
3. **Baseline** — `DummyRegressor(strategy="mean")` as a sanity-check reference.
4. **Feature selection** — `RFECV` wrapper using `RidgeCV` to drop low-value features.
5. **Train** — `GradientBoostingRegressor` and `XGBRegressor` on the selected feature subset.
6. **Evaluate & plot** — report metrics in both log and THB space, save predicted-vs-actual plots.
7. **Interpret** — SHAP-based global feature importance and summary plots for the GBDT.

`Shop Location` is encoded as **region-level** one-hot instead of province-level, reducing dimensionality while keeping the location signal.

> **Implementation note:** After adding `sentence-transformers`, the wrapper's `RFECV` and the XGBoost stage both run with `n_jobs=1`. Without this, the mix of torch/sentence-transformer threads and joblib parallelism caused a segfault during feature selection.

## Results

### Name-embedding + region-level model (current default)

| Step | Model | Features | R² (log) | RMSE (log) | MAE (log) | RMSE (THB) | MAE (THB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Mean predictor | — | −0.0001 | 0.9727 | 0.7766 | 638.08 | 348.71 |
| Selected GBDT | GradientBoostingRegressor | 66 | 0.2699 | 0.8311 | 0.6449 | 581.33 | 305.63 |
| Selected XGBoost | XGBRegressor | **66** | **0.2784** | **0.8262** | **0.6403** | **578.04** | **303.88** |

With the sentence-embedding features, **XGBoost now beats sklearn GradientBoostingRegressor** on the same selected subset. Both models improve substantially over the non-embedding region-level model.

### Region-level model without embeddings (previous default)

| Step | Model | Features | R² (log) | RMSE (log) | MAE (log) | RMSE (THB) | MAE (THB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Mean predictor | — | −0.0001 | 0.9727 | 0.7766 | 638.08 | 348.71 |
| Selected GBDT | GradientBoostingRegressor | **39** | **0.1658** | **0.8884** | **0.7013** | **607.32** | **326.31** |
| Selected XGBoost | XGBRegressor | 39 | 0.1598 | 0.8915 | 0.7033 | 607.60 | 326.48 |

On the selected 39 region-level features (no embeddings), sklearn GradientBoostingRegressor slightly outperforms XGBoost.

### Province-level model (earlier iteration)

| Step | Model | Features | R² (log) | RMSE (log) | MAE (log) | RMSE (THB) | MAE (THB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Mean predictor | — | −0.0001 | 0.9727 | 0.7766 | — | — |
| Full GBDT | GradientBoostingRegressor | 106 | 0.1963 | 0.8719 | 0.6895 | 607.78 | 322.80 |
| Selected GBDT | GradientBoostingRegressor | **99** | **0.2053** | **0.8671** | **0.6849** | **605.15** | **320.49** |

## What the wrapper cut

With the region + embedding encoding, `RFECV` kept **66 of 70** features. The four dropped features were:

| Dropped feature | Why it was likely cut |
|---|---|
| `has_reviews` | Review presence adds almost no price signal once `name_semantic_pca_*` and `name_word_count` are available. |
| `log_total_reviews` | Number of reviews is not a strong price predictor in this dataset. |
| `name_length` | Title character count is redundant with `name_word_count` and weaker than the semantic embeddings. |
| `shop_region_central` | The Central region one-hot carries less signal than other regions; the remaining region columns are enough. |

In short, the wrapper removed the review group, the weaker of the two title-length statistics, and the lowest-signal region column.

## Does `Shop Location` matter?

Yes — removing the entire location one-hot group hurts the model:

| Variant | Features | R² (log) | RMSE (log) | MAE (log) |
|---|---:|---:|---:|---:|
| Province-level GBDT | 99 | **0.2053** | **0.8671** | **0.6849** |
| Region-level GBDT | 39 | 0.1658 | 0.8884 | 0.7013 |
| Without any location encoding | ~33 | 0.1473 | 0.8982 | 0.7081 |
| Without `section_*` | ~13 | 0.1286 | 0.9080 | 0.7186 |

Both `Shop Location` (region) and `Section` carry useful signal. The best absolute performance comes from the province-level encoding, but the region-level encoding is a better **complexity/performance trade-off**.

## Region mapping

Provinces are grouped into the standard Thai geographic regions:

- `bangkok_metro` — Bangkok + Nonthaburi + Pathum Thani + Samut Prakan + Samut Sakhon + Nakhon Pathom
- `central` — Central provinces excluding metro
- `northern` — North
- `northeastern` — Isan/Northeast
- `eastern` — East
- `western` — West
- `southern` — South
- `overseas` — ต่างประเทศ
- `unknown` — Missing/unknown location

The mapping lives in `src/regions.py` and is used by both `src/feature_engineer.py` and `notebooks/03 feature_engineer.ipynb`.

## Model interpretation (SHAP)

`src/interpret.py` uses **TreeSHAP** on the selected GBDT to explain predictions. SHAP values answer: *"How much does each feature push this product's predicted log-price above or below the average prediction?"* They are derived from game theory, sum up to the model output, and work natively with tree ensembles.

Two plots are produced:

- **`shap_importance.png`** — global bar plot of mean |SHAP value| per feature. Higher bars mean the feature changes more predictions by a larger amount.
- **`shap_summary.png`** — beeswarm plot showing the distribution of SHAP values per feature. Color encodes the feature value (blue = low, red = high), and position shows direction/magnitude of the price effect.

### What is `name_semantic_pca_*`?

`name_semantic_pca_*` is the **meaning of the product name compressed into 32 numbers**.

- **name** = the product `Name` column (e.g. `"วิตามินซี 1000mg บำรุงผิว ของแท้"`).
- **semantic** = the *meaning* of the title, not just how long it is. We use `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) to read the Thai/English text and turn it into a 384-number vector that captures concepts like "vitamin", "premium", "imported", "herbal", etc.
- **PCA** = we then compress those 384 numbers down to 32 components so the tree model can handle them.

PCA components do **not** get human-readable labels like "premiumness". They are just ordered directions of variance: `pca_00` is the broadest direction, `pca_04` is the 5th direction, etc. The model happens to find `pca_04` most useful for predicting price, but that direction may combine several real-world concepts at once.

| Rank | Feature | Mean absolute SHAP value | Interpretation |
|---:|---|---:|---|
| 1 | `name_semantic_pca_04` | 0.0926 | 5th PCA direction of the product-name embedding. This learned semantic direction most strongly separates high/low log-price predictions in the model. |
| 2 | `name_semantic_pca_00` | 0.0881 | 1st PCA direction — usually captures the broadest semantic variance in product titles. |
| 3 | `name_word_count` | 0.0750 | Title length/wordiness remains a strong price signal even after adding embeddings — longer, more detailed titles correlate with higher/premium listings. |
| 4 | `name_semantic_pca_14` | 0.0705 | 15th PCA direction; a more specific semantic direction that still carries substantial pricing signal. |
| 5 | `section_Herbs_Traditional_Medicine` | 0.0583 | This category strongly shifts price downward — herbal/traditional products tend to be cheaper than average. |

### Location signal

Region features rank far below the embedding/title signals:

| Feature | Mean absolute SHAP value |
|---|---:|
| `shop_region_northeastern` | 0.0089 |
| `shop_region_northern` | 0.0063 |
| `shop_region_eastern` | 0.0036 |
| `shop_region_southern` | 0.0034 |
| `shop_region_unknown` | 0.0025 |

This confirms that **product title semantics carry much more pricing signal than seller location** after adding embeddings. Location still adds a small amount of explanatory power, but the dominant price drivers are now the PCA-compressed sentence-embedding representation of the product name and the product section.

### How to act on this

- **Product title semantics** is now the highest-leverage signal: the sentence-embedding components explain the largest share of SHAP value. A dedicated text model or larger embedding dimension could improve predictions further.
- **Title length still matters**: `name_word_count` remains in the top 3, so longer, keyword-rich titles continue to correlate with higher prices.
- **Section-level pricing** is the next strongest structured signal: each health/wellness category has its own price band.
- **Location is secondary**: region encoding is sufficient; province-level encoding only marginally improves performance at much higher dimensionality.

### Interpretation artifacts

| File | Location |
|---|---|
| SHAP feature importance bar plot | `notebooks/figures/shap_importance.png` |
| SHAP summary (beeswarm) plot | `notebooks/figures/shap_summary.png` |
| Predictions on raw table | `dataset/predictions.csv` |

## All artifacts

| File | Location |
|---|---|
| Feature engineering module | `src/feature_engineer.py` |
| Sentence-embedding module | `src/embeddings.py` |
| Region mapping | `src/regions.py` |
| Model interpretation | `src/interpret.py` |
| Feature-engineered table (with embeddings) | `dataset/health_and_wellness_feature_engineered.csv` |
| Feature-engineered table (without embeddings) | `dataset/health_and_wellness_feature_engineered_no_embeddings.csv` |
| GBDT model | `models/gradient_boosting.joblib` |
| XGBoost model | `models/xgboost.joblib` |
| Fitted name-embedding PCA | `models/name_pca.joblib` |
| Selected feature list | `models/selected_features.json` |
| GBDT result plot | `notebooks/figures/predicted_vs_actual_gbdt.png` |
| XGBoost result plot | `notebooks/figures/predicted_vs_actual_xgboost.png` |
| Province vs region comparison | `notebooks/figures/model_comparison.png` |
| GBDT vs XGBoost comparison | `notebooks/figures/gbdt_vs_xgboost.png` |
| With/without embeddings comparison | `notebooks/figures/embedding_comparison.png` |
| Baseline vs trained models | `notebooks/figures/baseline_comparison.png` |
| SHAP importance | `notebooks/figures/shap_importance.png` |
| SHAP summary | `notebooks/figures/shap_summary.png` |

All reproducible artifacts are ignored by git.

## Comparison plots

- `model_comparison.png` — province-level vs region-level GBDT.
- `gbdt_vs_xgboost.png` — GBDT vs XGBoost on the selected region-level features.
- `embedding_comparison.png` — GBDT and XGBoost with vs without sentence embeddings.
- `baseline_comparison.png` — baseline mean predictor vs GBDT vs XGBoost (R², RMSE log, RMSE THB).

All use small multiples or grouped bars so each metric uses its own natural scale.

## Outcome (basic)

The pipeline produces a usable price-prediction model with a single command:

```bash
uv run python main.py   # train + evaluate + interpret + compare
uv run python predict.py # generate predictions.csv from the raw table
```

Final model: **name-embedding + region-level XGBRegressor** on 66 selected features.

| Metric | Value |
|---|---:|
| R² (log) | 0.2784 |
| RMSE (log) | 0.8262 |
| MAE (log) | 0.6403 |
| RMSE (THB) | 578.04 |
| MAE (THB) | 303.88 |
| CV R² (log) | 0.2630 ± 0.0220 |

Prediction script: `predict.py` loads `models/xgboost.joblib` and `models/name_pca.joblib`, transforms a raw listing (`Name`, `Section`, `Shop Location`, `Total Reviews`) the same way as training, and returns predicted THB prices in `dataset/predictions.csv`.

## Performance work

A focused pass improved **both** runtime and accuracy capability. The deployed model is **unchanged**
unless a tuned model clearly beats it on the held-out test (per scope: add capability, don't auto-swap).

### Runtime

- **Embedding cache** (`models/name_embeddings.joblib`, keyed on a SHA-256 of the `Name` series, cap 8
  name-sets). Repeat `main.py` / `predict.py` runs on the same dataset skip the MiniLM encode entirely
  — the dominant cost.
- **PCA reuse = correctness fix + speed.** `predict.py` now loads the fitted `models/name_pca.joblib`
  and calls `pca.transform` instead of refitting on the prediction batch. This both removes the refit
  cost and fixes a latent train/predict basis mismatch: sklearn's randomized SVD gives
  `fit_transform(X) ≠ fit(X).transform(X)` by ~0.03, so the old predict path silently shifted features
  vs training. Training now uses the same `fit`-then-`transform` path as predict, so both are
  consistent (max abs diff ~7.6e-8) and recover the original R²≈0.267.
- **Predict fast path** (`build_predict_table` in `src/feature_engineer.py`) skips the target column
  and reuses the fitted PCA; batches missing some section/region dummies are padded via `reindex`.

### Accuracy capability

Shared `src/metrics.py` hoists the duplicated `evaluate()` and adds K-fold CV (mean±std). Run the
tuning capability with `uv run python -m src.tune`; results are written to `models/tuning_results.json`.

| Candidate | Held R² (log) | Held RMSE (THB) | CV R² (log) |
|---|---:|---:|---:|
| Current XGBoost (deployed) | 0.2784 | 578.04 | 0.2630 ± 0.0220 |
| Tuned XGBoost | 0.3381 | 544.51 | 0.3153 ± 0.0270 |
| Tuned GBDT | 0.3382 | 550.84 | 0.3034 ± 0.0233 |
| HistGradientBoosting | 0.3366 | 544.88 | 0.3016 ± 0.0206 |
| RandomForest | **0.3413** | 558.58 | 0.3039 ± 0.0201 |

All four tuned candidates are **clear winners** over the deployed XGBoost (each ΔR² ≥ +0.058),
with 32-dim embeddings lifting every candidate vs the prior 16-dim run (e.g. RandomForest
0.3186 → 0.3413). RandomForest stays the **clear top candidate** on held-out R² (ΔR² +0.063 vs
deployed) yet is **not** auto-promoted. To promote: `cp models/tuned_rf.joblib models/xgboost.joblib`
and re-run `predict.py` (selected-features list and PCA are unchanged within this run). Note the
`n_jobs=1` ceiling makes the RF search the slowest stage (~305s); it is the natural candidate for
subprocess-isolated parallelism next.

## What drives the price

1. **Product name semantics** — PCA-compressed sentence embeddings dominate SHAP importance.
2. **Title length (`name_word_count`)** — longer titles still correlate with higher prices.
3. **Product section** — each health/wellness category has its own price band.
4. **Shop region** — small effect compared to name/section.

## Next steps (not done)

- Promote the tuned RandomForest (`models/tuned_rf.joblib`, held R² 0.3413) over the deployed model
  once the uplift is confirmed on fresh data — capability exists, swap is opt-in (see Performance work).
- Try a Thai-specific sentence encoder (`EMBEDDING_DIM` is now 32; a different encoder would need a
  fresh embedding cache).
- Reclaim joblib parallelism for the tuning search via subprocess isolation (the `n_jobs=1` ceiling
  is deliberate — avoids the torch/joblib segfault on macOS; see `src/embeddings.py`).
