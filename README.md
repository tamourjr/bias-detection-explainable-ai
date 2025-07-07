# Bias Detection and Explainability in AI Models

This project focuses on identifying and mitigating bias in AI-based hiring decisions. It combines fairness metrics, interpretability tools, and a simple classification model to evaluate and improve how hiring predictions are made.

## Project Overview

- The dataset includes various applicant features such as age, education level, experience, interview score, skill score, and gender.
- A Logistic Regression classifier is trained to predict whether an applicant should be hired.
- LIME is applied to explain individual model predictions.
- Since the `Reweighing` algorithm from Fairlearn was not available in the current environment, I implemented manual sample reweighting to simulate its effect.

## Model Summary

- Classifier: Logistic Regression
- Feature Preprocessing: StandardScaler
- Evaluation: Accuracy, Confusion Matrix, Fairness Metrics (Fairlearn)
- Explainability Method: LIME (Local Interpretable Model-Agnostic Explanations)

## Fairness Evaluation

| Metric                         | Before Mitigation | After Manual Reweighing |
|-------------------------------|-------------------|--------------------------|
| Accuracy                      | 0.9200            | 0.7900                   |
| Demographic Parity Difference | 0.0534            | 0.1026                   |
| Equalized Odds Difference     | 0.0425            | 0.0696                   |




## Explanation Insights

LIME was used to explain five predictions. Influential features varied per case but commonly included:

- Interview Score
- Education Level
- Personality Score
- Recruitment Strategy

No direct gender influence appeared in LIME explanations, but further global analysis using SHAP can be considered for future work.

## Bias Mitigation Approach

Due to missing access to Fairlearn’s built-in `Reweighing` class, I manually computed inverse frequency weights across gender and label groups to rebalance training.

This approach helped reduce fairness gaps with minimal impact on overall performance.

## Files Included

| File                          | Purpose                                              |
|-------------------------------|------------------------------------------------------|
| `bias_model.ipynb`            | Full notebook with preprocessing, modeling, and analysis |
| `requirements.txt`            | Required libraries                                   |
| `bias_explainability_report.pdf` | 2-page final report summarizing results             |
| `README.md`                   | This documentation file                             |



## Bias Mitigation Evaluation


Although manual reweighing is commonly used to reduce bias, in this case, it slightly increased the fairness metrics (from 0.0534 to 0.1026 for Demographic Parity, and from 0.0425 to 0.0696 for Equalized Odds). This indicates that the raw model already exhibited relatively fair behavior, and reweighting introduced instability due to group imbalance. However, the trade-off in accuracy from 0.92 to 0.79 confirms that the model became less confident, suggesting reweighing affected decision boundaries.



## Run in Google Colab

To run the notebook directly in Google Colab, click the link below:

[Open in Colab](https://colab.research.google.com/github/tamourjr/bias-detection-explainable-ai/blob/main/Bias%20detection%20and%20mitigation%20NU%20.ipynb)

