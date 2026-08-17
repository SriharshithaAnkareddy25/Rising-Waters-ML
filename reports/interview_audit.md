# Interview audit

## A. Original system

The original project loaded 115 rows from Excel, capped selected outliers using bounds calculated on the full dataset, made a non-stratified 80/20 split, standardized all ten features, and trained several classifiers in notebooks. A class-weighted Logistic Regression pipeline was serialized for a Flask form. The notebook's Logistic Regression result was 100% on a 23-row test set containing three floods; no reproducible 98% result was found.

## B. Problems discovered

- `Jun-Sep > 2400` reproduces all 115 labels, making label construction or same-period leakage the primary concern.
- `ANNUAL` is the seasonal rainfall sum to within 0.1 mm and is redundant.
- Outlier caps were learned before splitting, leaking test-distribution information.
- Only 16 rows are positive; majority accuracy is already 86.09%.
- The original test set had only three positive rows and the split was not explicitly stratified.
- Random Forest training accuracy was computed from Decision Tree predictions.
- XGBoost training and test accuracy were computed from KNN predictions.
- Several stochastic models had no fixed seed; no CV, baseline, tuning boundary, or ROC-AUC was used.
- The saved artifacts had incompatible scikit-learn serialization versions.
- Flask manually duplicated feature ordering, exposed raw exceptions, accepted implausible values, ran in debug mode, and described uncalibrated scores as literal flood chances.
- Dataset provenance, time period, geography, label definition, and the meanings of `avgjune` and `sub` are absent.

## C. Changes implemented

- The primary experiment excludes `ANNUAL` and `Jun-Sep` and records why. This is a defensible ablation, not proof that every remaining feature is leakage-free.
- A fixed, stratified 80/20 split preserves an untouched final test set. Five-fold shuffled stratified CV operates only on the 92-row training partition.
- Every estimator receives the same folds and named features through scikit-learn pipelines and a `ColumnTransformer`.
- Logistic Regression is scaled; tree models are not. No global outlier capping is used because the extremes may be genuine flood-related observations.
- Dummy majority, Logistic Regression, Decision Tree, Random Forest, and XGBoost are compared with accuracy, precision, recall, F1, ROC-AUC, confusion matrices, training time, and inference time.
- Balanced weights are used where supported. SMOTE was rejected because there was no evidence that synthetic points from only 13 training positives would improve validity.
- Only Logistic Regression and XGBoost were tuned with bounded grids. Search and selection use training folds only.
- Final selection prioritizes recall because false negatives are more dangerous, with F1 and ROC-AUC as tie-breakers.
- EDA, correlations, ROC curves, confusion matrix, feature importance, and logistic coefficients are generated reproducibly.
- Flask loads one fitted pipeline, constructs a named DataFrame, enforces a schema and observed ranges, handles bad input with HTTP 400, and displays a safety disclaimer.
- Tests cover schema validation, missing/unexpected fields, numeric/range validation, preprocessing shape, artifact loading, prediction, and Flask routes.

## D. Final measured results

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy majority | 0.859064 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.869565 | 0.000000 | 0.000000 | 0.000000 | 0.500000 |
| Logistic Regression | 0.728655 | 0.338333 | 0.633333 | 0.417778 | 0.795972 | 0.826087 | 0.400000 | 0.666667 | 0.500000 | 0.683333 |
| Decision Tree | 0.728655 | 0.196667 | 0.400000 | 0.257143 | 0.592500 | 0.782609 | 0.000000 | 0.000000 | 0.000000 | 0.450000 |
| Random Forest | 0.836842 | 0.000000 | 0.000000 | 0.000000 | 0.781389 | 0.826087 | 0.000000 | 0.000000 | 0.000000 | 0.700000 |
| XGBoost | 0.824561 | 0.506667 | 0.300000 | 0.323810 | 0.714861 | 0.869565 | 0.500000 | 0.666667 | 0.571429 | 0.700000 |
| Logistic Regression (tuned) | 0.771930 | 0.369048 | 0.700000 | 0.460476 | 0.810972 | 0.826087 | 0.400000 | 0.666667 | 0.500000 | 0.650000 |
| XGBoost (tuned) | 0.815205 | 0.503333 | 0.533333 | 0.480952 | 0.685694 | 0.869565 | 0.500000 | 0.666667 | 0.571429 | 0.716667 |

The selected model's test confusion matrix is `[[17, 3], [1, 2]]`. Timing measurements are available in `model_comparison.csv` but are hardware-dependent and should not be presented as universal benchmarks.

## E. Final model decision

Tuned Logistic Regression (`C=0.01`, balanced class weights) was selected because its mean CV recall was 0.700000, higher than every alternative, and because it is simple and interpretable. Tuned XGBoost had higher test F1, but selecting it after seeing the test result would misuse the holdout. Logistic Regression missed one of three test floods and raised three false alerts. This is not adequate for operational use, but it is an honest result.

## F. Difficult interview questions

1. What evidence suggests `Jun-Sep` is target leakage rather than merely predictive?
2. Why did you exclude both `Jun-Sep` and `ANNUAL`, and what information might still leak through the remaining features?
3. Why is the original perfect test result weak evidence when the test set contains three positive cases?
4. Why use Logistic Regression for a possibly nonlinear environmental process?
5. Why was accuracy insufficient, and how did the DummyClassifier demonstrate that?
6. What did the selected confusion matrix reveal that accuracy hid?
7. Why prioritize recall, and what operational damage can poor precision cause?
8. Why use stratification for both the holdout and CV folds?
9. What is cross-validation estimating, and why does it not replace an untouched test set?
10. How did the pipeline prevent preprocessing leakage?
11. Why scale Logistic Regression but not Decision Trees, Random Forest, or XGBoost?
12. Why did Random Forest have reasonable ROC-AUC while predicting no positives at threshold 0.5?
13. What is the difference between ranking quality measured by ROC-AUC and thresholded recall/F1?
14. Why not automatically lower the decision threshold to increase recall?
15. Why did you avoid SMOTE with only 13 positive training examples?
16. How does Random Forest reduce the variance and overfitting of one Decision Tree?
17. How does boosting in XGBoost differ from independent bagged trees in Random Forest?
18. Why did tuned XGBoost not become the final model despite better holdout metrics?
19. Did tuning improve generalization, and which numbers support that conclusion?
20. What do standardized Logistic Regression coefficients mean, and why are they not causal effects?
21. Why is `ANNUAL` plus all seasonal components a multicollinearity problem?
22. How would you redesign the dataset for a true forecast made before monsoon season?
23. Why would a temporal split be preferable if years and dates were available?
24. How would you obtain confidence intervals with so few positive observations?
25. Are the output scores calibrated probabilities? How would you test and calibrate them?
26. What forms of geographic, sensor, and climate distribution shift could occur?
27. What would you monitor in production: input drift, missingness, output rate, calibration, delayed recall, or alert volume?
28. How would you respond if the model's positive prediction rate doubled without a corresponding weather change?
29. What safety fallback should exist when sensors fail or inputs fall outside training ranges?
30. What ethical and legal limitations apply to presenting this as a public flood alert?

## G. Verified resume facts

- Audited a 115-row binary rainfall dataset with 10 original features and identified a single-feature threshold that reproduced 100% of supplied labels.
- Designed a leakage-aware eight-feature ablation and reproducible stratified train/test plus five-fold cross-validation workflow.
- Compared a majority baseline, Logistic Regression, Decision Tree, Random Forest, and XGBoost using accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and timing.
- Performed bounded GridSearchCV tuning for Logistic Regression and XGBoost using training folds only.
- Selected tuned class-weighted Logistic Regression based on mean CV recall of 0.700000; measured untouched-test recall 0.666667 and F1 0.500000 on a three-positive test set.
- Generated class-distribution, feature-distribution, correlation, ROC, confusion-matrix, feature-importance, and coefficient visualizations.
- Deployed the fitted preprocessing/model pipeline through Flask with named-feature validation, range checks, graceful malformed-input handling, and probability-score disclosure.
- Added automated tests for schema enforcement, preprocessing shape, artifact loading, prediction, and Flask inference.
