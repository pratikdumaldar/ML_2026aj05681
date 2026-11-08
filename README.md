# ML_2026aj05681

# ML Classification Assignment

## Problem Statement
The objective of this project is to build an end-to-end machine learning pipeline that trains five different classification models on a dataset, evaluates their performance using multiple metrics, and deploys an interactive frontend using Streamlit. The Streamlit app relies on pre-trained model artifacts rather than retraining models on the fly.

## Dataset Description
**Breast Cancer Wisconsin (Diagnostic) Dataset**
* **Instances:** 569
* **Features:** 30 numeric, predictive attributes (e.g., radius, texture, perimeter, area, smoothness).
* **Target:** Binary classification (Malignant vs. Benign).

## Links
* **GitHub Repository:** https://github.com/pratikdumaldar/ML_2026aj05681
* **Streamlit App URL:** https://ml2026aj05681.streamlit.app/

## Metrics Comparison Table
| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9561 | 0.9977 | 0.9583 | 0.9718 | 0.9650 | 0.9064 |
| Decision Tree | 0.9473 | 0.9473 | 0.9577 | 0.9577 | 0.9577 | 0.8872 |
| KNN | 0.9561 | 0.9856 | 0.9583 | 0.9718 | 0.9650 | 0.9064 |
| Naive Bayes | 0.9736 | 0.9984 | 0.9594 | 1.0000 | 0.9793 | 0.9439 |
| SVM | 0.9473 | 0.9974 | 0.9577 | 0.9577 | 0.9577 | 0.8872 |

## Observations
* **Best Performer:** Naive Bayes performed exceptionally well on this dataset, achieving the highest accuracy and a perfect recall score, making it highly effective for medical diagnostic scenarios where minimizing false negatives is critical.
* **Consistency:** Logistic Regression and SVM also showed robust AUC scores (>0.99), indicating strong capability in separating the classes.
* **Speed vs. Accuracy:** The Decision Tree trained the fastest but had slightly lower AUC compared to the probabilistic models, reflecting standard trade-offs in tree-based algorithms without ensemble methods.