# 🏏 IPL 50+ Score Prediction System (EDA → Machine Learning → Deployment)

## 📌 Project Overview

This project started as a **Data Analytics / EDA project** on IPL batting data and was later upgraded into a **full end-to-end Machine Learning system**.

🎯 **Goal:**  
Predict whether a batsman will score **50+ runs** in an innings based on early/mid-innings match context.

---

## 🎯 Target Variable

high_score = 1 → Batsman scores 50+ runs
high_score = 0 → Batsman does NOT score 50+ runs


---

## 📊 Dataset

- Rows: ~15,720  
- Original columns: Match, player, innings, and performance data  
- Final ML features (clean & realistic):

ballsfaced
minutes
runningscore
runningover
isnotout
captain
current_innings
innings_id


---

## 🧹 Phase 1: Data Cleaning & EDA

- Removed missing & invalid values  
- Fixed data types  
- Standardized column names  
- Generated summary tables  
- Visualized:
  - Top batsmen by runs
  - Strike rate leaders
  - Team-wise & season-wise trends
- Exported cleaned dataset

---

## 🛠️ Feature Engineering

- Created target column:
```python
high_score = (runs >= 50)
Removed data leakage columns:

runs, fours, sixes, strikerate, etc.

Selected only deployable numeric features
```
--- 
## 🤖 Models Trained

Logistic Regression (baseline)

Random Forest

🏆 Gradient Boosting (Best Model)

---

##  📈 Evaluation Metrics

Precision

Recall

F1-score (main metric due to class imbalance)

ROC-AUC

Confusion Matrix

---

## 🏆 Results Summary
| Model                | F1-score  | ROC-AUC   |
| -------------------- | --------- | --------- |
| Logistic Regression  | ~0.77     | ~0.98     |
| Random Forest        | ~0.74     | ~0.97     |
| 🏆 Gradient Boosting | **~0.80** | **~0.98** |

---

## 🚀 Running the Application

### Prerequisites
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

### Run the Streamlit App
```bash
streamlit run app.py
```

The app provides:
- **Home**: Overview of the project
- **Data Exploration**: View data, statistics, and visualizations
- **Feature Engineering**: See how features are prepared
- **Modeling**: Train and evaluate models
- **Prediction**: Make predictions for 50+ runs

---

## 🧠 Key ML Concepts Used

End-to-end ML workflow

Feature engineering

Data leakage prevention

Train-test split

StandardScaler

Model comparison

Gradient Boosting

F1-score for imbalanced classification

Pipeline-based training

Model deployment

---

## 👨‍💻 Author

Rahul Singh
BE Artificial Intelligence & Data Science
India 🇮🇳

---