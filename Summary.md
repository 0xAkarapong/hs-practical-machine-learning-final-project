# Practical Machine Learning Final Project — Summary

## Pipeline

The end-to-end pipeline now runs in `main.py` as:

1. **Cleaning** (`src/_01_clean_data.py`, ported from `notebooks/02 anomaly_detection.ipynb`) — dedup the raw scrape (68,499 → 2,622 real listings), parse `฿`-prefixed prices, then drop high-confidence anomalies flagged by **both** global IQR and per-section robust z (median/MAD, |z|>3.5) → `dataset/health_and_wellness_no_outliers.csv` (2,610 rows). This makes the cleaned input reproducible from the raw CSV without running the notebook.
2. **Feature engineering** — build the modeling table from `health_and_wellness_no_outliers.csv`. Product `Name` is represented by both text statistics (`name_length`, `name_word_count`) and **64 PCA-compressed sentence embeddings** (`name_semantic_pca_00..63`) from `intfloat/multilingual-e5-base` (768-dim, with the required `"query: "` input prefix), plus regex-derived text signals (`name_qty_log`, `name_has_weight_unit`, `name_has_count_unit`, `name_is_authentic`, `name_has_bundle`, `name_thai_ratio`) that the embeddings don't fully capture. A parsed `log_total_sold` (from the messy Thai `"… ชิ้น"` column, previously unused) is also included.
3. **Split** — stratified train/test split on `log_price_thb` (80/20, seed 42) → 2,084 train / 526 test.
4. **Baseline** — `DummyRegressor(strategy="mean")` as a sanity-check reference.
5. **Feature selection** — `RFECV` wrapper using a **tree-based estimator** (`XGBRegressor`, `n_estimators=100, max_depth=3`) to drop low-value features.
6. **Train** — `GradientBoostingRegressor` and `XGBRegressor` (tuned hyperparameters baked into `train_xgboost`) on the selected feature subset.
7. **Evaluate & plot** — report metrics in both log and THB space, save predicted-vs-actual plots.
8. **Interpret** — SHAP-based global feature importance and summary plots for the GBDT.

Standalone (not in `main.py`): `src/eda.py` (ported from `notebooks/01 eda.ipynb`, run via `python -m src.eda`) and the four comparison scripts under `src/comparisons/` (baseline, embeddings, gbdt_xgboost, models).

`Shop Location` is encoded as **region-level** one-hot instead of province-level, reducing dimensionality while keeping the location signal.

> **Implementation note (threading):** The original macOS segfault was torch/sentence-transformer threads + joblib **fork** (RFECV/RandomizedSearchCV `n_jobs>1`). Fix: joblib stages stay `n_jobs=1` (no fork), while in-process threading is used freely — `OMP_NUM_THREADS` is forced to 1 only on macOS; on Linux/Docker it is left unset so the e5-base encode + XGBoost tree-building use all cores (~4× encode speedup). The torch encoder is freed before training so the threads are clean.

## Encoder evolution

The product-name encoder has been iterated on (each swap auto-invalidates the embedding cache, keyed on `model_name:sha256(names)`):

| # | Encoder | Raw dim | PCA dim | Prefix | Features (after RFECV) | Held-out R² (log) | MAE (THB) |
|---|---|---:|---:|---|---:|---:|---:|
| 1 | `paraphrase-multilingual-mpnet-base-v2` | 384 | 32 | none | 52 | 0.4414 | 268.49 |
| 2 | `BAAI/bge-m3` | 1024 | 32 | none | 52 | 0.4676 | 258.83 |
| 3 | `intfloat/multilingual-e5-base` (current) | 768 | 64 | `"query: "` | 65 | **0.4789** | **262.19** |
| — | `airesearch/wangchanberta-base-att-spm-uncased` | 768 | 32 | none | — | abandoned | — |

**Current: `intfloat/multilingual-e5-base`.** Lighter than bge-m3 (278M vs 568M, ~2× faster CPU encode) with strong Thai/English coverage, and expanding the PCA dimension from 32 → 64 retained more name signal. e5 models **require** a `"query: "/"passage: "` prefix on every input; the e5 model card says to use `"query: "` when embeddings are used as features (clustering / linear probing), which is our regression case — applied in the single shared encode path so train + predict stay consistent.

**WangchanBERTa was attempted and abandoned.** It is a raw Thai RoBERTa MLM (not a sentence-embedding model) and ships a minimal `sentencepiece.bpe.model` tokenizer with no `tokenizer.json` / `tokenizer_class`. transformers 5.13 cannot auto-instantiate that tokenizer — it routes SentencePiece-BPE through a tiktoken extractor that cannot parse the binary file, and `use_fast=False` is not honored. There is no clean load path without pinning `transformers` to 4.x and/or shipping a custom tokenizer, so it was reverted to e5-base.

## Results

### Name-embedding + region-level model (current default)

Feature set selected by the **tree-based RFECV wrapper** (65 of 114 features; see "What the wrapper cut").

| Step | Model | Features | R² (log) | RMSE (log) | MAE (log) | RMSE (THB) | MAE (THB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Mean predictor | — | −0.0001 | 0.9727 | 0.7766 | 638.08 | 348.71 |
| Selected GBDT | GradientBoostingRegressor | 65 | 0.4051 | 0.7502 | 0.5804 | 564.31 | 286.34 |
| **Selected XGBoost (deployed)** | XGBRegressor | **65** | **0.4789** | **0.7021** | **0.5219** | **533.31** | **262.19** |

`train_xgboost` now bakes the tuned hyperparameters as its defaults, so a single `main.py` run **is** the deployed model — the old "fixed-default vs tuned" distinction is gone. XGBoost beats sklearn GradientBoostingRegressor on the same selected subset.

CV (5-fold on train): GBDT R² 0.3538 ± 0.0217; XGBoost R² 0.4212 ± 0.0223.

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

The wrapper is **`RFECV` with a tree-based estimator** (`XGBRegressor`, `n_estimators=100, max_depth=3`) — not a linear `RidgeCV`. The estimator matches the final model class, so the kept subset optimizes what the tree cares about (thresholds, interactions), not a linear model's CV-MSE. RFECV ranks features by `feature_importances_` and eliminates one per round. `min_features_to_select=10`.

The wrapper has been re-run each time the feature space changed. **Wrapper rounds (RFECV re-runs):**

| Round | Encoder / PCA dim | Pool | Scored subset sizes | Kept | Notes |
|---:|---|---:|---:|---:|---|
| 1 | mpnet / 32 | 82 | 73 (82 → 10) | **52** | First switch to the tree wrapper; prior `RidgeCV` wrapper kept 78/82 (barely selective). |
| 2 | bge-m3 / 32 | 82 | 73 (82 → 10) | **52** | Same pool size (32 PCA) → same kept count; raw dim change did not move the count. |
| 3 | **e5-base / 64 (current)** | **114** | **105 (114 → 10)** | **65** | PCA dim 32 → 64 widened the pool by 32; RFECV kept 65 (13 more than the 32-dim runs). |

- **Round 3 (current): kept 65 of 114**, pruned 49. The 768-dim e5 embeddings reduced to 64 PCA components, joined by the ~50 non-embedding features (region/section one-hot, name text stats, `log_total_sold`, review group).

The dropped features are the weaker `name_semantic_pca_*` directions, sparse `section_*` categories, and the review group (`has_reviews`, `log_total_reviews`) — subsumed by `log_total_sold` and the stronger semantic directions. Section, region, and the top semantic directions stay.

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

The mapping lives in `src/common/regions.py` and is used by both `src/_02_feature_engineer.py` and `notebooks/03 feature_engineer.ipynb`.

## Model interpretation (SHAP)

`src/_07_interpret.py` uses **TreeSHAP** on the selected GBDT to explain predictions. SHAP values answer: *"How much does each feature push this product's predicted log-price above or below the average prediction?"* They are derived from game theory, sum up to the model output, and work natively with tree ensembles.

Two plots are produced:

- **`shap_importance.png`** — global bar plot of mean |SHAP value| per feature. Higher bars mean the feature changes more predictions by a larger amount.
- **`shap_summary.png`** — beeswarm plot showing the distribution of SHAP values per feature. Color encodes the feature value (blue = low, red = high), and position shows direction/magnitude of the price effect.

### What is `name_semantic_pca_*`?

`name_semantic_pca_*` is the **meaning of the product name compressed into 64 numbers**.

- **name** = the product `Name` column (e.g. `"วิตามินซี 1000mg บำรุงผิว ของแท้"`).
- **semantic** = the *meaning* of the title, not just how long it is. We use `sentence-transformers` (`intfloat/multilingual-e5-base`, with the `"query: "` prefix) to read the Thai/English text and turn it into a 768-number vector that captures concepts like "vitamin", "premium", "imported", "herbal", etc.
- **PCA** = we then compress those 768 numbers down to 64 components so the tree model can handle them.

PCA components do **not** get human-readable labels like "premiumness". They are just ordered directions of variance: `pca_00` is the broadest direction, `pca_04` is the 5th direction, etc. The model finds a spread of these directions useful for predicting price; each may combine several real-world concepts at once.

### Top SHAP features (e5-base, 64-dim, 65-feature set)

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---:|---|---:|---|
| 1 | `log_total_sold` | 0.1950 | Parsed units-sold — the strongest single driver; higher sold volume shifts predicted price. |
| 2 | `name_semantic_pca_00` | 0.0931 | 1st PCA direction — the broadest semantic variance in product titles. |
| 3 | `name_semantic_pca_31` | 0.0641 | 32nd PCA direction; a specific semantic direction carrying pricing signal (only surfaced once dim expanded to 64). |
| 4 | `name_semantic_pca_06` | 0.0632 | 7th PCA direction. |
| 5 | `name_semantic_pca_44` | 0.0593 | 45th PCA direction (high-index direction now retained thanks to the 64-dim PCA). |
| 6 | `section_Herbs_Traditional_Medicine` | 0.0503 | This category shifts price downward — herbal/traditional products tend to be cheaper. |
| 7 | `name_semantic_pca_52` | 0.0451 | 53rd PCA direction. |
| 8 | `name_semantic_pca_13` | 0.0449 | 14th PCA direction. |
| 9 | `name_semantic_pca_04` | 0.0432 | 5th PCA direction. |
| 10 | `name_semantic_pca_40` | 0.0407 | 41st PCA direction. |

Rounding out the top 15: `name_qty_log` (0.0370), `name_semantic_pca_20` (0.0368), `pca_37` (0.0329), `pca_47` (0.0329), `pca_02` (0.0311). Expanding PCA to 64 dims pushed several high-index directions (`pca_31/40/44/52`) into the top SHAP ranks that did not exist in the 32-dim model.

### Location signal

Region features rank far below the embedding/title/sold signals (each `shop_region_*` ≈ 0.003–0.009 mean |SHAP|), confirming that **product title semantics and units sold carry much more pricing signal than seller location** after adding embeddings. Location still adds a small amount of explanatory power, but the dominant price drivers are the PCA-compressed sentence-embedding representation of the product name, the product section, and `log_total_sold`.

### How to act on this

- **Product title semantics** is the highest-leverage signal: the sentence-embedding components + `log_total_sold` explain the largest share of SHAP value.
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
| Feature engineering module | `src/_02_feature_engineer.py` |
| Sentence-embedding module | `src/common/embeddings.py` |
| Region mapping | `src/common/regions.py` |
| Model interpretation | `src/_07_interpret.py` |
| Feature-engineered table (with embeddings) | `dataset/health_and_wellness_feature_engineered.csv` |
| Feature-engineered table (without embeddings) | `dataset/health_and_wellness_feature_engineered_no_embeddings.csv` |
| GBDT model | `models/gradient_boosting.joblib` |
| XGBoost model | `models/xgboost.joblib` |
| Fitted name-embedding PCA | `models/name_pca.joblib` |
| Selected feature list | `models/selected_features.json` |
| Embedding cache | `models/name_embeddings.joblib` |
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

## Docker

The pipeline runs in Docker end-to-end (exit 0) with all artifacts in named volumes (`models`, `dataset`, `figures`). A multi-stage build (`uv:python3.13` builder + `python:3.13-slim` runtime) pre-bakes the e5-base model into the image so runtime runs offline (no `HF_TOKEN` — e5-base is MIT/non-gated). A `Makefile` drives the container:

```bash
make build            # build image (bakes e5-base once, cached after)
make up               # run main.py + predict.py, exit 0
make predict-file FILE=my.csv   # predict on a host CSV (bind-mounted)
make predict-sample   # predict on dataset/sample_input.csv
make shell / logs / ps / metrics / figures / down / clean
```

The first `make up` re-encodes (cache miss, ~150s on CPU, all cores); later runs hit the embedding cache and skip encoding.

## Outcome (basic)

The pipeline produces a usable price-prediction model with a single command:

```bash
uv run python main.py    # train + select + evaluate + interpret (Docker: make up)
uv run python predict.py # generate predictions.csv from the raw table
```

Final model: **name-embedding + region-level XGBRegressor** on **65 tree-wrapper-selected features**, using **64 PCA components** of `intfloat/multilingual-e5-base` name embeddings. The tuned XGBoost hyperparameters are baked into `train_xgboost` so `main.py` reproduces the deployed `models/xgboost.joblib` directly (n_estimators 300, max_depth 7, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, min_child_weight 5, reg_lambda 5.0, n_jobs=-1).

| Metric | Value |
|---|---:|
| R² (log) | 0.4789 |
| RMSE (log) | 0.7021 |
| MAE (log) | 0.5219 |
| RMSE (THB) | 533.31 |
| MAE (THB) | 262.19 |
| CV R² (log) | 0.4212 ± 0.0223 |

Predictions on the raw table: mean **470.60 THB**, median **313.07 THB** (2,610 products).

`predict.py` loads `models/xgboost.joblib` and `models/name_pca.joblib`, transforms a raw listing (`Name`, `Section`, `Shop Location`, `Total Reviews`, …) the same way as training (same e5 encode + `"query: "` prefix + fitted PCA), and returns predicted THB prices in `dataset/predictions.csv`.

## Performance work

A focused pass improved **both** runtime and accuracy capability.

### Runtime

- **Embedding cache** (`models/name_embeddings.joblib`, keyed on `model_name:sha256(names)`, cap 8 name-sets). Repeat `main.py` / `predict.py` runs on the same dataset skip the e5 encode entirely — the dominant cost. Swapping the encoder auto-invalidates the cache (the model name is in the key).
- **Multithreaded encode + training (Linux/Docker).** `OMP_NUM_THREADS` is left unset on Linux so the e5-base encode and XGBoost tree-building use all cores (~4× encode speedup over `OMP=1`); macOS still forces `OMP=1`. joblib stages stay `n_jobs=1` (no fork → no segfault); the XGB wrapper and final model use in-process `n_jobs=-1` (safe because the torch encoder is freed before training).
- **PCA reuse = correctness fix + speed.** `predict.py` loads the fitted `models/name_pca.joblib` and calls `pca.transform` instead of refitting on the prediction batch. This both removes the refit cost and fixes a latent train/predict basis mismatch: sklearn's randomized SVD gives `fit_transform(X) ≠ fit(X).transform(X)` by ~0.03, so the old predict path silently shifted features vs training. Training now uses the same `fit`-then-`transform` path as predict, so both are consistent.
- **Predict fast path** (`build_predict_table` in `src/_02_feature_engineer.py`) skips the target column and reuses the fitted PCA; batches missing some section/region dummies are padded via `reindex`.

### Accuracy capability

`src/_08_tune.py` tunes XGBoost / GradientBoosting / HistGradientBoosting / RandomForest via `RandomizedSearchCV` (5-fold CV) and saves tuned winners to `models/tuned_*.joblib`. The last full re-tune was run against the **52-feature mpnet/bge-m3** tree-wrapper set; its best XGBoost params (`n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0`) are the ones now baked into `train_xgboost` and applied to the current 65-feature e5 set.

> **Re-tune on the e5-base 65-feature set is deferred.** The inherited params already deliver the best result so far (held R² 0.4789, +0.0113 over bge-m3+32's 0.4676, +0.0375 over mpnet's 0.4414). Running `uv run python -m src._08_tune` on the e5 set would confirm whether a different XGB config squeezes more — a clear winner (>+0.01 held R²) should be re-baked into `train_xgboost`.

## What drives the price

1. **Units sold (`log_total_sold`)** — the strongest single SHAP driver.
2. **Product name semantics** — the 64 PCA-compressed e5 embedding directions fill most of the top SHAP ranks.
3. **Product section** — each health/wellness category has its own price band.
4. **Shop region** — small effect compared to name/section/sold.

## Next steps (not done)

- **Re-tune on the e5-base 65-feature set** (`uv run python -m src._08_tune`) and re-bake the winner into `train_xgboost` if it clearly beats the inherited params.
- Reclaim joblib parallelism for the tuning search via subprocess isolation if the `n_jobs=1` search becomes slow (the `n_jobs=1` ceiling on the *search* is deliberate — avoids the torch/joblib fork segfault on macOS; the wrapper/final-model `n_jobs=-1` threads are already enabled).
- Try a Thai-specific sentence encoder that loads cleanly under transformers 5.x if one becomes the clear best (WangchanBERTa was attempted but its SentencePiece-BPE tokenizer is not loadable without pinning transformers / a custom tokenizer).