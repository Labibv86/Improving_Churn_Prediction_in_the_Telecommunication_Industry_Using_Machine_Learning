# Improving Churn Prediction in the Telecommunication Industry Using Machine Learning

A machine learning study on telecom customer churn prediction using the **Cell2Cell** benchmark dataset, with a systematic literature review, careful preprocessing, and a comparative evaluation of seven classifiers under realistic cross validation.

---

## Authors

| Name | Affiliation |
|------|-------------|
| Md. Shihab Sharar | Department of CSE, BRAC University, Dhaka, Bangladesh |
| Moazzem Hossain Majumder | Department of CSE, BRAC University, Dhaka, Bangladesh |
| Fahim Montasir | Department of CSE, BRAC University, Dhaka, Bangladesh |
| Nafisa Islam | Department of CSE, BRAC University, Dhaka, Bangladesh |
| Sabbir Ahmed | Department of CSE, BRAC University, Dhaka, Bangladesh |
| Md. Golam Rabiul Alam | Department of CSE, BRAC University, Dhaka, Bangladesh |

---

## Abstract

Acquiring a new telecom customer costs far more than retaining an existing one, which makes churn prediction a recurring business problem. This paper builds a churn prediction pipeline on the **Cell2Cell** dataset and reviews twenty related research articles covering Cell2Cell specific studies, other public and private telecom datasets, and two systematic reviews. Reported accuracies on Cell2Cell range from the fifties to nearly ninety nine percent, yet very few papers report model calibration or significance tests between competing models. Customer service and usage volatility features are repeatedly cited as important but are rarely combined into a composite feature.

---

## Problem Statement

A churn model must handle class imbalance, remain robust to out of training window data, and ideally rank customers by how valuable it is to spend retention resources on them. This project addresses those requirements on a standard benchmark while explicitly acknowledging that Cell2Cell is a de identified US postpaid wireless dataset, not a Bangladeshi one. The Bangladesh context only informs feature design, since it is a prepaid multi carrier market where switching costs are near zero.

---

## Literature Review Highlights

- **Cell2Cell studies:** Reported performance varies widely on the same 71,000 row benchmark, from 58.37 percent accuracy for a regularized ANN to 88.16 percent for a multi stage feature selection pipeline with SHAP and LIME explanations. One study reports AUC near 0.98 to 0.99, which warrants a leakage check before being reused as a benchmark.
- **Other telecom datasets:** Work on proprietary SyriaTel data integrated social network features into XGBoost for a 0.933 AUC with temporal evaluation, a practice absent from the Cell2Cell papers. A profit based metric called e Profits ranked classifiers differently than AUC or F1 alone.
- **Reviews:** A survey of 212 papers from 2015 to 2023 found no universally winning classifier or feature set, a prevalence of outdated public data, and a lack of profit sensitivity in evaluation. A PRISMA based review of 61 studies confirmed XGBoost and LightGBM dominance, growing use of LSTM and CNN models, and class imbalance, explainability, and concept drift as open problems.
- **Research gap:** Few studies report calibration, significance tests, or leakage controlled benchmarks, and volatility based composite features are underused.

---

## Dataset

- **Source:** Cell2Cell, originally from the Teradata Center for Customer Relationship Management at Duke University, available via Kaggle.
- **Size:** 51,047 customer records with 58 attributes, including the churn label.
- **Class distribution:** About 71 percent loyal customers and 29 percent churners, roughly a 2.5 to 1 imbalance.
- **Features:** Monthly charges, minutes used, overage minutes, dropped and blocked calls, age, income, marital status, device information, and more.

---

## Exploratory Data Analysis

- **Missing values:** 11 columns contain missing data. AgeHH1 and AgeHH2 have the most at about 1,002 each, percentage change columns have about 396, and ServiceArea has 23.
- **High correlations:** Three pairs exceed 0.8, namely MonthlyMinutes with ReceivedCalls at 0.834, UniqueSubs with ActiveSubs at 0.811, and Handsets with HandsetModels at 0.895. These redundant variables were later removed.
- **Distributions:** MonthlyRevenue ranges from negative values to 1,223 with a median of 48.46, MonthlyMinutes ranges from 0 to 7,359 with a median of 366, and OverageMinutes is strongly right skewed with a median of 3 and outliers up to 4,321.
- **Categorical variables:** ServiceArea has 748 categories, far too many to use directly. Other categorical attributes are binary, such as 75.8 percent of customers having no children at home and 90.2 percent having mobile internet capability.

---

## Data Cleaning and Preprocessing

- **Split:** An 80/20 stratified train and holdout test split with 56,837 training and 14,210 test observations, keeping the churn ratio identical and preventing test leakage.
- **Outliers:** Numerical outliers capped at the 99th percentile computed from the training set only.
- **Imputation:** KNN imputation with five neighbors for demographic features, median imputation for numeric features, and mode imputation for categorical features. HandsetPrice was converted to numeric before imputation.
- **Rare categories:** ServiceArea reduced from 748 categories to 15, grouping everything below 1 percent frequency into an Other category.
- **Feature sets:** Two datasets were produced, one with all 57 variables and one with 54 variables after removing ReceivedCalls, UniqueSubs, and HandsetModels.
- **Scaling:** StandardScaler for numeric features and appropriate encoding for categorical features.

---

## Models Evaluated

Seven baseline classifiers were compared:

1. **Logistic Regression** with lbfgs solver, 1000 max iterations, L2 penalty with C equal to 1.0, and balanced class weights.
2. **Random Forest** with 30 estimators, unlimited depth, min samples split of 2, and min samples leaf of 1.
3. **AdaBoost** with 50 estimators and a learning rate of 1.0.
4. **XGBoost** with 50 estimators, max depth 6, learning rate 0.3, subsample and colsample by tree of 0.8, and a scale positive weight of 3.8.
5. **CatBoost** with 100 iterations, depth 6, learning rate 0.3, L2 regularization of 3.0, and balanced auto class weights.
6. **MLP Neural Network** with two hidden layers of 8 and 2 neurons, ReLU activation, adam solver, alpha of 0.001, learning rate 0.003, and early stopping with a validation fraction of 0.1 and patience of 10.
7. **Linear SVC** with L2 regularization, C equal to 1.0, squared hinge loss, and balanced class weights.

---

## Class Imbalance and Evaluation Methodology

- **Strategies:** Models were trained both without balancing and with SMOTE applied inside each cross validation fold. XGBoost and CatBoost used their built in class weighting.
- **Validation:** Stratified 5 fold cross validation preserves class distribution in every fold, which is essential for imbalanced classification.
- **Final testing:** The best model from each group was retrained on the full training set and evaluated on the untouched holdout test set.
- **Metrics:** Accuracy, precision, recall, F1 score, and ROC AUC, with confusion matrices and classification reports.

---

## Results

### Ablation Study

Eight feature groups were tested. **Equipment Features** were the strongest group for most models, including MLP, CatBoost, XGBoost, and AdaBoost. Call Quality Features gave the best cross validation F1 for SVC, while Service Features were most effective for Random Forest. Equipment data is the strongest overall churn signal, and ensemble trees and neural networks capture its non linearities best.

### Final Test Set Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| AdaBoost | 0.5110 | 0.2519 | 0.6912 | **0.3692** | **0.6005** |
| CatBoost | 0.5046 | 0.2501 | 0.6969 | 0.3680 | 0.5988 |
| MLP (NN) | 0.4914 | 0.2462 | **0.7065** | 0.3652 | 0.5899 |
| XGBoost | 0.5446 | 0.2477 | 0.5892 | 0.3488 | 0.5833 |
| Logistic Regression | **0.5774** | 0.2552 | 0.5428 | 0.3472 | 0.5867 |
| Random Forest | 0.4943 | 0.2349 | 0.6391 | 0.3435 | 0.5745 |
| SVC (RBF) | 0.3291 | 0.2059 | 0.7847 | 0.3263 | 0.5090 |

**Key findings:**
- AdaBoost and CatBoost achieved the best F1 scores with recalls near 0.70, detecting roughly 70 percent of churners.
- MLP gave a competitive F1 of 0.3652 and the second highest recall of 0.7065.
- Logistic Regression had the highest accuracy of 0.5774 but a lower F1 than the boosting models.
- SVC had the highest recall of 0.7847 but the lowest precision of 0.2059, producing many false positives.
- Recall matters more than accuracy in a retention context, since missing a churner is costlier than a false alarm.

---

## Discussion

Related works claim F1 scores up to 88 percent and AUC above 95 percent, but this study's leakage controlled cross validation and holdout testing produced a realistic best F1 of 0.3692. Even contemporary ANN architectures did not improve on this. The gap suggests that some published results may depend on optimistic sampling or test set reuse, and that rigorous evaluation reveals the true real world performance.

---

## Conclusion

Seven machine learning algorithms were compared for telecom churn prediction. Ablation testing across eight feature groups showed that Equipment Features have the greatest impact on churn for both ensemble trees and neural networks. AdaBoost was the best overall model with an F1 of 0.3692 and ROC AUC of 0.6005, while CatBoost and MLP were close behind. Although Logistic Regression had the highest accuracy, boosting models were superior in recall, which is the critical metric for retention. Good recall came at the cost of precision, producing many false positives.

**Future work** includes synthetic oversampling with SMOTE to better handle class imbalance and sequential modeling with RNNs or Transformers to capture temporal churn behavior.

---

## Citation

If you reference this work, please cite the paper as:

> M. S. Sharar, M. H. Majumder, F. Montasir, N. Islam, S. Ahmed, and M. G. R. Alam, "Improving Churn Prediction in the Telecommunication Industry Using Machine Learning," Department of Computer Science and Engineering, BRAC University, Dhaka, Bangladesh.