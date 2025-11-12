# 📊 Bank Marketing Classification – Comparing Machine Learning Models

## 🧠 Project Overview
This project applies machine learning techniques to the **Bank Marketing Dataset** from the UCI Machine Learning Repository.  
The objective is to predict whether a client will subscribe to a term deposit after being contacted by a marketing campaign.

The work follows the **CRISP-DM process** (Business Understanding → Data Preparation → Modeling → Evaluation) and compares four supervised classifiers:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Support Vector Machine (SVM)**

Additional experiments explored **hyperparameter tuning**, **class imbalance handling**, and **metric optimization** using **F1-score** and **ROC-AUC**.

---

## 📂 Files
| File | Description |
|------|--------------|
| `prompt_III.ipynb` | Main Jupyter Notebook containing all 11 problems from the assignment, including EDA, model building, tuning, and evaluation. |
| `bank-additional-full.csv` | Dataset used for training and testing models (UCI Bank Marketing dataset). |
| `CRISP-DM-BANK.pdf` | Reference paper describing dataset background and prior analysis. |
| `README.md` | Project summary and documentation (this file). |

---

## ⚙️ Technical Details
- **Language:** Python (Jupyter Notebook)
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn  
- **Environment:** Python 3.10+  
- **Data Split:** 70% training / 30% testing (stratified)

---

## 📈 Models Compared

| Model | Train Accuracy | Test Accuracy | F1 (positive class) | ROC-AUC |
|-------|----------------|----------------|--------------------|----------|
| Logistic Regression | 0.8999 | **0.9022** | 0.49 | 0.80 |
| KNN | 0.9122 | 0.8916 | 0.47 | 0.79 |
| Decision Tree | 0.9962 | 0.8382 | 0.46 | 0.76 |
| SVM | 0.8972 | 0.8982 | 0.48 | 0.79 |
| Random Forest (Tuned) | — | 0.883 | **0.52** | **0.81** |

---

## 💡 Key Insights
- The dataset is **imbalanced** (~90% “no” vs 10% “yes” responses).  
- **Logistic Regression** achieved the highest accuracy and generalization.  
- **Random Forest (tuned)** and **balanced Logistic Regression** improved **F1** and **recall** for the minority (positive) class.  
- **Decision Trees** showed signs of overfitting; **SVM** provided strong ROC-AUC but longer training time.  
- Optimizing for **F1** instead of accuracy led to better business outcomes (capturing more true “yes” clients).

---

## 🎯 Business Impact
By identifying clients most likely to subscribe to term deposits, the model enables:
- More efficient targeting for future campaigns  
- Reduction in unnecessary calls  
- Improved return on marketing investment (ROI)

---

## 🚀 Next Steps
1. Add **ensemble models** (XGBoost / Gradient Boosting) for potentially higher F1/ROC-AUC.
2. Implement **cost-sensitive thresholding** to balance call cost vs. conversion probability.
3. Deploy as a simple API or Streamlit dashboard for campaign managers.

---

## 👩‍💻 Author
**[Your Name]**  
Machine Learning Capstone – Assignment 17.1  
*(Bank Marketing Classification – CRISP-DM Implementation)*  

