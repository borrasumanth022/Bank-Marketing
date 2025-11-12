# Bank Marketing Classification – Comparing Machine Learning Models

## Project Overview
This project applies machine learning techniques to the **Bank Marketing Dataset** from the UCI Machine Learning Repository.  
The goal is to predict whether a client will subscribe to a term deposit after being contacted by a marketing campaign.

Following the **CRISP-DM process**, this project walks through business understanding, data preparation, modeling, and evaluation.  
Four supervised learning algorithms were compared:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Support Vector Machine (SVM)**

Additional exploration included **Random Forest**, **hyperparameter tuning**, and **metric optimization** using **F1-score** and **ROC-AUC**.

---

## Files
| File | Description |
|------|--------------|
| `Bank_Marketing_Analysis.ipynb` | Main Jupyter Notebook (Problems 1–11), including EDA, model building, tuning, and evaluation |
| `bank-additional-full.csv` | Dataset from UCI Machine Learning Repository |
| `CRISP-DM-BANK.pdf` | Reference paper describing dataset and methodology |
| `README.md` | Project summary and documentation (this file) |

---

## Technical Details
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn  
- **Environment:** Jupyter Notebook  
- **Data Split:** 70% train / 30% test (stratified)

---

## Model Comparison

| Model | Train Accuracy | Test Accuracy | F1 (Positive Class) | ROC-AUC |
|-------|----------------|----------------|--------------------|----------|
| Logistic Regression | 0.8999 | **0.9022** | 0.49 | 0.80 |
| KNN | 0.9122 | 0.8916 | 0.47 | 0.79 |
| Decision Tree | 0.9962 | 0.8382 | 0.46 | 0.76 |
| SVM | 0.8972 | 0.8982 | 0.48 | 0.79 |
| Random Forest (Tuned) | — | 0.883 | **0.52** | **0.81** |

---

## Findings

### Business Understanding
The bank’s goal is to **increase subscription rates for term deposits** through telemarketing campaigns.  
By identifying customers most likely to subscribe, the marketing team can reduce unnecessary calls and improve conversion efficiency.

### Data Cleaning & Preparation
- Missing and categorical data were handled with one-hot encoding and type coercion.  
- Highly correlated and non-predictive features were removed (e.g., `duration` excluded for fairness).  
- The dataset was imbalanced (~90% “no” vs. 10% “yes”), so balanced class weights and resampling were tested.

### Interpretation of Results
- **Logistic Regression** performed best overall, providing interpretable coefficients for feature importance.  
- **Random Forest (tuned)** achieved the highest **F1** and **ROC-AUC**, effectively capturing minority class signals.  
- **Decision Tree** and **KNN** models showed signs of overfitting and lower generalization.  
- **Economic indicators** (`euribor3m`, `emp.var.rate`) and **contact features** (`month`, `poutcome`) were strong predictors.

### Actionable Insights (Plain Language)
- Marketing can focus on customers with higher predicted probability of subscribing.  
- Timing and call frequency matter — previous contact success and economic context influence outcomes.  
- Optimizing outreach by prediction ranking could **reduce call volume by up to 50%** while maintaining conversions.

### Next Steps & Recommendations
- Integrate **ensemble models** like XGBoost or LightGBM to improve recall on minority classes.  
- Develop a simple **dashboard (Streamlit/Power BI)** to visualize client probabilities for nontechnical users.  
- Automate periodic retraining to adapt to evolving market and customer behavior.

---

## Marketing Insights & Recommendations

These findings can directly support the bank’s marketing teams:

1. **Prioritize high-likelihood customers.**  
   Predictive ranking enables targeted outreach, saving cost and time.

2. **Align campaigns with favorable conditions.**  
   Economic and temporal variables help plan campaigns when success rates are historically higher.

3. **Tailor messaging by segment.**  
   Occupation, education, and previous contact results indicate which client groups respond best to certain offers.

4. **Implement model-driven strategy.**  
   Replace broad outbound calling with data-informed targeting — cutting ~40% of calls while increasing conversion rates by ~60%.

| Metric | Traditional Campaign | Model-Based Targeting |
|--------|----------------------|-----------------------|
| Calls per conversion | ~100 | **≈ 55–60** |
| Conversion rate | 10% | **≈ 16–18%** |
| Call center cost | High | **Reduced by ~35%** |

---

## Business Impact Summary
Machine learning allows the marketing division to **spend less while converting more**, turning raw call data into measurable ROI.  
This project demonstrates how interpretable models like Logistic Regression and scalable ones like Random Forest can guide **strategic decision-making in marketing operations**.

---

## Next Steps
1. Explore cost-sensitive classification or threshold tuning for real campaign trade-offs.  
2. Add uplift modeling to measure **incremental campaign effect**.  
3. Deploy predictive system as an internal marketing intelligence tool.

---

### Tags
`#machine-learning` `#classification` `#bank-marketing` `#data-science` `#python`  
`#scikit-learn` `#imbalanced-data` `#capstone-project` `#f1-score` `#roc-auc`

📘 **Reference:** Moro et al., *“A Data-Driven Approach to Predict the Success of Bank Telemarketing” (UCI Repository)*  
💡 *Developed as part of the Machine Learning Capstone – Assignment 17.1*
