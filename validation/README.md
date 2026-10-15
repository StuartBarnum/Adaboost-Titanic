# Validation review — October 15, 2026

Reviewed baseline commit: `02bab74774a61c08bf54a211e9d900e655bea6ce`.
The original `.R`, rendered `.md`, input data and README are unchanged.

## Findings

1. **Outcome-conditioned age imputation is not deployable as fitted.** In each outer fold, observed training ages are averaged within Title/Pclass/**Survived**. Missing training ages therefore depend on their own labels. Held-out labels are correctly masked first, so this is not direct access to the held-out answer. Rather, it creates a training/prediction mismatch: missing held-out ages are averaged only among the masked passengers, which in LOO means Kaggle test passengers with the same title/class. The held-out passenger's observed covariates also join this transductive preprocessing. This procedure does not justify concluding that accuracy is inflated; the direction requires empirical comparison.

2. **Restricting surname target means does not remove self-inclusion.** A training row's outcome contributes to the surname mean whenever that feature is enabled. In a surname group of n labeled training passengers, flipping its outcome changes its own feature by 1/n. At the code's actual schedule (`index %% 5 == 2`, not every third iteration as the comment says), separate models are fitted to passengers with and without this feature. Restricting the schedule or restricting to surnames in the prediction batch cannot make those training features independent of their own labels. Surname membership in the unlabeled Kaggle test batch also changes which training rows/models are used. This makes the original prediction rule batch-dependent.

3. **The outer holdout is otherwise real.** Each labeled passenger is excluded from model fitting and masked before feature aggregation. Kaggle test labels are unavailable. The prediction matrix takes the first row of a PassengerId-sorted prediction batch; that happens to select the held-out passenger with these files (training IDs 1–891, Kaggle test IDs 892–1309). Revised evaluation matches by ID explicitly. Accuracy lists inside the original model contain NA because most test outcomes are NA; they are not used to fit the model or in the final outer accuracy calculation. Confusion-matrix argument names in the original final loop are reversed (`pred` contains truth), although this does not change accuracy.

4. **The validation target matters.** Passenger LOO allows known outcomes for relatives of a held-out passenger. That is appropriate only for predicting another passenger when such outcomes would really be available. It does not estimate generalization to unseen families or to prospective rescue decisions. Surname alone is an imperfect family identifier. The grouped sensitivity analysis holds out connected components sharing surname OR ticket, conservatively separating both; common surnames can merge unrelated people. Randomized five-group allocation is not stratified and group sizes can differ. Inspect the saved folds and class counts before interpretation.

5. **Model selection remains a separate source of optimism.** The published 60-round accuracy curve peaks at 0.8271605 (737/891) at round 46, compared with 0.7979798 (711/891) at round 1. These are historical outputs, not new results. Reporting the maximum after examining this curve, depths, costs, bins or other settings uses the validation data for selection. Its optimism cannot be estimated from that curve alone. Use a fixed round count decided before rerunning (the new runner defaults to 10), or tune all such choices within an inner CV loop and evaluate once in untouched outer folds. The 10-round result is still a retrospective sensitivity analysis on a familiar dataset, not a fresh confirmatory estimate.

6. **Boosting and reproducibility issues.** The original uniformly bootstraps rows then applies adaptive case weights; this is a stochastic CART-boosting variant. No RNG seed/parallel RNG stream is set, so its exact historical output cannot be reproduced reliably. Unhandled weighted errors of 0 or 1 cause infinite weights; chance-or-worse learners can receive zero/negative weights. The revised learner preserves uniform bootstrap plus weighted CART, but handles perfect learners, rejects chance-or-worse learners, and uses stable log-weight updates. It disables unused rpart internal CV and avoids storing every fitted tree. These changes mean the revised-versus-original contrast is a whole-procedure comparison, not a pure estimate of leakage bias.

## Targeted alternatives

`leakage_free.R` contains separate, testable preprocessing and prediction functions:

- **no_target (primary):** training-only mean age by title/class, then class, then overall mean; the same map is applied to fitting and held-out rows. Surname outcome means are removed. Surname counts use fitting covariates only. Unseen factor levels become missing for CART to handle. No held-out label is passed to prediction.
- **honest_target:** before any fitting, randomly reserve 30% of outer-training passengers as a reference set. Compute surname outcome means exclusively from that reference and apply the identical map to the disjoint learner-fitting subset and held-out passengers. Unseen surnames use reference prevalence. This avoids both self-inclusion and bootstrap duplicates of a passenger reintroducing its label. Reference labels must remain available at deployment. A single reference split costs training information; its result should not be interpreted as an optimized solution.
- **honest_no_target:** the exact same reduced fitting subset, seed and learner without target encoding. Compare this with honest_target to assess whether surname outcomes add predictive information under an honest encoding scheme. Repeat prespecified seeds to assess reference-split variation; never select the best seed.
- **age_only_ablation:** change only the original age grouping to exclude Survived. It retains original transduction, surname self-inclusion and learner behavior; it is diagnostic, not a leakage-free alternative.

Do not simply subtract one's outcome from a surname mean and call the result fully safe: group encodings then share labels across rows and bootstrap duplicates require special care. The disjoint reference construction avoids this ambiguity. No family target encoding is needed for the primary recommendation.

## Execution and tests

From the repository root, with R and the listed packages installed:

```r
install.packages(c("rpart", "tidyverse", "stringi", "stringr"))
```

```sh
Rscript validation/test_validation.R
Rscript validation/run_review.R smoke 3 20261015
Rscript validation/run_review.R loo 10 20261015
Rscript validation/run_review.R grouped 10 20261015
# Optional historical-curve diagnostic; do not use its maximum as unbiased accuracy:
Rscript validation/run_review.R loo 60 20261015
```

Tests check held-out-label poisoning, age independence from training labels, test-covariate isolation, exact age fallback values, encoding independence from learner-fitting labels, overlap rejection, deterministic predictions and degenerate-class behavior. The runner executes five methods on common folds, saves passenger predictions and fold IDs in RDS, fixed-round accuracy summaries, paired gained/lost counts and accuracy changes, all errors and session information. Failed predictions remain missing; compare methods on their common successful passengers and require full coverage before reporting an overall result. Smoke mode uses the first ten passengers only and is a plumbing check, not an accuracy estimate.

For the legacy run only, replace vector `&&` by `&` in memory for modern dplyr compatibility. This affects the unused Surname_Train_Test diagnostic field, not predictors. The original file is untouched. Loading functions explicitly avoids accidentally launching the original parallel main program.

**Execution status:** This environment has no R/Rscript. Installing R was attempted, but package index downloads failed and no installable R package became available. Consequently the revised R code and its tests have NOT been executed here, and no new predictive accuracy or paired delta is reported. Static review and a Python audit of the supplied CSVs were completed. `feature_audit.json` records 891 labeled passengers, 418 unlabeled passengers, 177 missing labeled ages, 534 passengers with unique training surnames, and 357 with a training surname peer. These are data diagnostics, not performance evidence. The original script's SHA-256 is recorded for preservation checks.

## What remains supported

The repository remains a substantive custom statistical-programming demonstration. The published passenger-LOO results are evidence about the historical procedure under its particular transductive setup; they do not establish leakage-free accuracy or an independently validated gain from family survival information. There is currently insufficient evidence to say that removing the problematic transformations improves or reduces accuracy, or that the family hypothesis survives honest evaluation. Run the supplied tests and comparisons before updating any performance claim. For uncertainty in comparisons involving family outcomes, use group-aware resampling and ideally repeated full validation; a naive independent-passenger binomial interval understates dependence. Descriptive paired gained/lost counts are supplied without an independence-based significance claim.
