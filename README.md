# ML-Classification-Streamlit-App
Machine Learning Classification Models with Streamlit Deployment
# Machine Learning Classification using Streamlit

## 1. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models to predict breast cancer diagnosis. The application allows users to upload test data, select a model, and visualize performance metrics.

---

## 2. Dataset Description

The Breast Cancer dataset contains medical attributes computed from digitized images of breast mass cell nuclei.

- Total Instances: 569
- Number of Features: 30
- Target Variable: target (0 = Benign, 1 = Malignant)

---

## 3. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes Classifier
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

---

## 4. Model Evaluation Metrics

The following metrics were used:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5. Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.93859649122807 | 0.923190304618408 | 0.973684210526315 | 0.860465116279069 | 0.91358024691358 | 0.870222910252124 |
| Decision Tree | 0.929824561403508 | 0.925319358008516 | 0.906976744186046 | 0.906976744186046 | 0.906976744186046 | 0.850638716017032 |
| KNN | 0.75438596491228 | 0.711103832296102 | 0.741935483870967 | 0.534883720930232 | 0.621621621621621 | 0.45990657885545 |
| Naive Bayes | 0.614035087719298 | 0.492957746478873 | 0 | 0 | 0 | -0.0732092287582023 |
| Random Forest | 0.964912280701754 | 0.95807402554864 | 0.975609756097561 | 0.930232558139534 | 0.952380952380952 | 0.925285392066775 |
| XGBoost | 0.956140350877193 | 0.951031772027513 | 0.952380952380952 | 0.930232558139534 | 0.941176470588235 | 0.90637859429323 |

---

## 6. Observations on Model Performance

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Performs well due to linear separability. |
| Decision Tree | May overfit on training data. |
| KNN | Sensitive to feature scaling. |
| Naive Bayes | Lower performance due to independence assumption. |
| Random Forest | Strong performance due to ensemble learning. |
| XGBoost | High accuracy with balanced performance. |

---

