# EasyVisa – Visa Certification Prediction

## Project Overview

This project applies machine learning to support the visa certification process by identifying applicant and employer characteristics associated with visa approval outcomes.

The objective was to develop a classification model that could help streamline application review and support consistent, data-driven decision-making.

## Business Problem

The growing volume of visa applications creates a need for analytical tools that can assist reviewers in identifying applications with higher or lower likelihoods of certification.

The project therefore focused on:

- analysing applicant and employer characteristics
- identifying important drivers of visa certification
- comparing classification models
- addressing class imbalance
- selecting a model that balances false approvals and false denials
- translating model results into practical decision-support recommendations

## Data

The dataset contained information on both applicants and employers, including:

- applicant continent
- education level
- previous job experience
- job-training requirement
- employment type
- prevailing wage
- employer size
- employer year of establishment
- visa certification outcome

## Analytical Workflow

The project included:

1. Data inspection and cleaning
2. Exploratory Data Analysis
3. Univariate and bivariate analysis
4. Data preprocessing
5. Classification modelling
6. Class-imbalance treatment
7. Model comparison
8. Hyperparameter tuning
9. Model evaluation using classification metrics
10. Business interpretation and recommendations

## Final Model

The final recommended approach was a **tuned Random Forest classifier trained on oversampled data**.

The model was evaluated with emphasis on the **F1-score**, helping balance the risks associated with both false approvals and false denials.

The model was designed as a **decision-support tool rather than an automatic decision-maker**.

## Key Insights

The analysis indicated that visa certification outcomes were influenced by both applicant and employer characteristics.

Important factors included:

- education level
- previous job experience
- full-time employment
- prevailing wage
- employer size
- employer maturity

Applicants with stronger qualifications and employment profiles generally showed higher certification rates.

## Skills Demonstrated

- Python
- Pandas
- NumPy
- Exploratory Data Analysis
- Data Preprocessing
- Classification
- Random Forest
- Ensemble Learning
- Oversampling
- Hyperparameter Tuning
- Model Evaluation
- Business Analytics
- Decision-Support Modelling

## Responsible Use

The predictive model should support human review rather than replace visa decision-makers. Borderline or higher-risk cases should remain subject to appropriate manual assessment.

## Author

**Joseph Kayefor Ebigwai**

Data Science Portfolio Project
