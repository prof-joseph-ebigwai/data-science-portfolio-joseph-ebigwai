### 1. [Loan Default Prediction](./loan-default-prediction)

## Project Overview

This project develops a machine-learning approach for identifying borrowers who may be at elevated risk of loan default or severe delinquency.

The objective was not simply to maximise predictive accuracy, but to develop a practical credit-risk decision-support solution capable of identifying higher-risk applications while avoiding unnecessary restriction of satisfactory borrowers.

## Dataset

The analysis used historical information from 5,960 previously approved loan applicants.

The dataset included borrower characteristics, credit history, repayment capacity, collateral information and historical loan outcomes.

## Project Workflow

The project included:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Statistical validation of exploratory findings
- Feature engineering
- Training, validation and independent test splitting
- Class-imbalance analysis
- Model comparison
- Hyperparameter tuning
- Operating-threshold analysis
- Model interpretation
- Independent test evaluation
- Translation of model results into business recommendations

## Machine-Learning Models Evaluated

The following classification algorithms were investigated:

- Logistic Regression
- K-Nearest Neighbours
- Decision Tree
- Random Forest
- Gradient Boosting

Different approaches to class imbalance were also evaluated, including random oversampling, random undersampling and class-weighted modelling.

## Final Model

The preferred solution was a tuned Gradient Boosting model trained using an oversampling strategy.

An operating threshold of 0.34 was selected for enhanced review rather than automatic rejection.

On the independent test dataset, the model identified approximately 81.5% of historical defaults.

The project therefore demonstrates how predictive analytics can support risk-based review while retaining human credit oversight.

## Technologies and Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook
- Statistical analysis
- Machine-learning model evaluation

## Skills Demonstrated

- Data preparation
- Exploratory data analysis
- Statistical testing
- Feature engineering
- Predictive modelling
- Binary classification
- Class-imbalance management
- Hyperparameter optimisation
- Precision-recall analysis
- Threshold optimisation
- Model validation
- Business interpretation of machine-learning outputs

## Business Application

The final model was designed as a decision-support system rather than an automatic loan-rejection engine.

Higher-risk applications can be directed for enhanced review while lower-risk applications continue through normal underwriting processes.

This approach demonstrates how machine learning can help financial institutions prioritise credit-review resources and improve consistency in risk assessment.

## Author

**Joseph Kayefor Ebigwai**

Data Science Portfolio  
Email: ebijoe4@gmail.com
