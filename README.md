# Optimizing Bank Marketing Campaigns with Machine Learning

This project applies **Machine Learning** to improve the effectiveness of bank marketing campaigns, specifically in predicting whether a client will subscribe to a term deposit.

## 📌 Project Overview
Banks often run marketing campaigns to encourage clients to subscribe to term deposit products. However, traditional approaches can be inefficient — contacting a large number of people with a low conversion rate wastes time and resources.  
This project leverages historical campaign data to build a predictive model that can identify potential subscribers more effectively.

## 🎯 Objectives
- Analyze past marketing data to uncover key customer patterns.
- Build a predictive model to determine the likelihood of a client subscribing.
- Optimize the campaign strategy to improve success rates while reducing costs.

## 📂 Dataset
The dataset used is **Bank Marketing Campaign Data**, containing customer profiles, previous interactions, and campaign details.  
Notable columns include:
- `age`, `job`, `marital`, `education` (customer demographics)
- `balance`, `housing`, `loan` (financial information)
- `contact`, `month`, `poutcome` (communication details)
- `y` (target variable: yes/no)

## 🔍 Exploratory Data Analysis (EDA)
Key findings:
- Certain age groups (30–40) have higher subscription rates.
- Previous campaign success is a strong predictor of future subscription.
- Contact month significantly affects the likelihood of success.

## 🛠 Machine Learning Approach
We tested multiple algorithms, including:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

**Random Forest** was selected as the final model for its stability and strong performance.

## 📈 Model Optimization
- Initial threshold (0.5) gave decent precision but low recall, missing many potential customers.
- Adjusting the decision threshold to **0.35** improved recall from ~49% to ~71%, boosting estimated net benefit to **$342,768**.

## 📊 Business Impact
By targeting only the most likely subscribers:
- The marketing team can save resources by avoiding low-probability contacts.
- Campaign success rate increases, improving ROI.

## 🚀 Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn, XGBoost, Imbalanced-learn
- Jupyter Notebook

## 📌 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/akmalfalah/bank-marketing-ml.git
   cd bank-marketing-ml

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bank-marketing-campaign-prediction-jahbpghh88f8s4d7uzk5kg.streamlit.app/)

