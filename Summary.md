# Practical Machine Learning Final Project — Summary

## Pipeline

The end-to-end pipeline now runs in `main.py` as:

1. **Feature engineering** — build the modeling table from `health_and_wellness_no_outliers.csv`.
2. **Split** — stratified train/test split on `log_price_thb` (80/20, seed 42).
3. **Baseline** — `DummyRegressor(strategy="mean")` as a sanity-check reference.
4. **Feature selection** — `RFECV` wrapper using `RidgeCV` to drop low-value features.
5. **Train** — `GradientBoostingRegressor` and `XGBRegressor` on the selected feature subset.
6. **Evaluate & plot** — report metrics in both log and THB space, save predicted-vs-actual plots.
7. **Interpret** — SHAP-based global feature importance and summary plots for the GBDT.

`Shop Location` is encoded as **region-level** one-hot instead of province-level, reducing dimensionality while keeping the location signal.

## Results

### Region-level model (current default)

| Step | Model | Features | R² (log) | RMSE (log) | MAE (log) | RMSE (THB) | MAE (THB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Mean predictor | — | −0.0001 | 0.9727 | 0.7766 | — | — |
| Full GBDT | GradientBoostingRegressor | 43 | 0.1861 | 0.8775 | 0.6909 | 607.05 | 323.77 |
| Selected GBDT | GradientBoostingRegressor | **39** | **0.1658** | **0.8884** | **0.7013** | **607.32** | **326.31** |
| Selected XGBoost | XGBRegressor | 39 | 0.1598 | 0.8915 | 0.7033 | 607.60 | 326.48 |

On the selected 39 region-level features, **sklearn GradientBoostingRegressor outperforms XGBoost**. A grid search on the same 39 features found XGBoost's best R² = 0.1523 (`learning_rate=0.03, max_depth=7, n_estimators=100`), still below GBDT.

### Province-level model (previous iteration)

| Step | Model | Features | R² (log) | RMSE (log) | MAE (log) | RMSE (THB) | MAE (THB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Mean predictor | — | −0.0001 | 0.9727 | 0.7766 | — | — |
| Full GBDT | GradientBoostingRegressor | 106 | 0.1963 | 0.8719 | 0.6895 | 607.78 | 322.80 |
| Selected GBDT | GradientBoostingRegressor | **99** | **0.2053** | **0.8671** | **0.6849** | **605.15** | **320.49** |

## What the wrapper cut

With the region encoding, `RFECV` kept **39 of 43** features. The four dropped columns are usually the review/name numeric features plus one or two low-signal region columns.

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

### Top 5 features by mean |SHAP value|

| Rank | Feature | Mean |SHAP value| | Interpretation |
|---:|---|---:|---|
| 1 | `name_word_count` | 0.1066 | Product title length/wordiness is the strongest price signal. Longer, more detailed titles are associated with higher (or more premium) listings. |
| 2 | `section_Herbs_Traditional_Medicine` | 0.0732 | This category strongly shifts price downward — herbal/traditional products tend to be cheaper than average. |
| 3 | `section_Protein` | 0.0482 | Protein supplements command distinct, often higher pricing. |
| 4 | `section_Skin_Nourishment` | 0.0405 | Skincare-related products carry a category-specific price premium. |
| 5 | `section_Breast_Enlargement` | 0.0382 | A small niche category with its own distinct price level. |

### Location signal

Region features rank lower than section/name signals:

| Feature | Mean |SHAP value| |
|---|---:|
| `shop_region_northern` | 0.0206 |
| `shop_region_northeastern` | 0.0179 |
| `shop_region_eastern` | 0.0133 |

This confirms that **product section and title text carry more pricing signal than seller location**, but location still contributes meaningfully to the model. The regional effect is likely capturing differences in shipping costs, local competition, or supply-chain tiers.

### How to act on this

- **Product title optimization** is the highest-leverage signal: longer, keyword-rich titles correlate with price. A title-embedding or text-length model could improve predictions further.
- **Section-level pricing** is also strong: each health/wellness category has its own price band. A section-aware model or separate per-section baselines could help.
- **Location is secondary**: region encoding is sufficient; province-level encoding only marginally improves performance at much higher dimensionality.

### Interpretation artifacts

| File | Location |
|---|---|
| SHAP feature importance bar plot | `notebooks/figures/shap_importance.png` |
| SHAP summary (beeswarm) plot | `notebooks/figures/shap_summary.png` |

## All artifacts

| File | Location |
|---|---|
| Feature engineering module | `src/feature_engineer.py` |
| Region mapping | `src/regions.py` |
| Model interpretation | `src/interpret.py` |
| Feature-engineered table | `dataset/health_and_wellness_feature_engineered.csv` |
| GBDT model | `models/gradient_boosting.joblib` |
| XGBoost model | `models/xgboost.joblib` |
| Selected feature list | `models/selected_features.json` |
| GBDT result plot | `notebooks/figures/predicted_vs_actual_gbdt.png` |
| XGBoost result plot | `notebooks/figures/predicted_vs_actual_xgboost.png` |
| Province vs region comparison | `notebooks/figures/model_comparison.png` |
| GBDT vs XGBoost comparison | `notebooks/figures/gbdt_vs_xgboost.png` |
| SHAP importance | `notebooks/figures/shap_importance.png` |
| SHAP summary | `notebooks/figures/shap_summary.png` |

All reproducible artifacts are ignored by git.

## Comparison plots

- `model_comparison.png` — province-level vs region-level GBDT.
- `gbdt_vs_xgboost.png` — GBDT vs XGBoost on the selected region-level features.

Both use small multiples (R² and RMSE on separate subplots) so each metric uses its own natural scale.

## Recommendation

Use the **region-level GradientBoostingRegressor** as the default pipeline:

- It uses only **39 features** vs 99 for province-level.
- It beats XGBoost on the same feature set.
- THB error is nearly identical to the province-level model (607 vs 605 RMSE).

Keep the province-level model as evidence that finer location granularity helps slightly if complexity is not a constraint. XGBoost is not worth switching to here unless much more tuning or larger data is introduced.

## Next steps (not done)

- Hyperparameter tuning for the GBDT itself (not just XGBoost).
- Try `HistGradientBoostingRegressor` or `RandomForestRegressor`.
- Engineer richer text features from product `Name`.
- Add cross-validation beyond the single held-out test set.
