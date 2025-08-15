# random-forest-xgboost-telemarketing-campaign
Machine learning models for telemarketing compaign – Random Forest and XGBoost implementation from my Master's Thesis.

# Master's Thesis – Predicting Customer Response in Bank Marketing Campaigns using Random Forest and XGBoost

This repository contains the code and selected materials from my Master's Thesis,  
which focused on applying machine learning models (**Random Forest** and **XGBoost**)  
to predict customer response in a Portuguese bank’s telemarketing campaign  
([Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)).

The main objective was to evaluate the effectiveness of these algorithms in a **binary classification** problem,  
with a particular focus on how **data balancing techniques** (SMOTE, class weighting, data reduction)  
influence model performance on an **imbalanced dataset**.

## Key findings
- cross-validation was applied in all experiments — **AUC > 70%** for every model and dataset variant.
- In direct comparisons, **XGBoost** consistently outperformed Random Forest across all dataset configurations (unbalanced, reduced, SMOTE-balanced, weighted).
- **Data reduction** significantly improved sensitivity (better at identifying positive cases).  
- **SMOTE** improved precision (fewer false positives).  
- Class weighting improved results for XGBoost, but worsened them for Random Forest.  
- The best approach depends on campaign goals — maximizing reach (sensitivity) vs. reducing false positives (precision).  

## Context & motivation
The problem of **class imbalance** is common in marketing campaigns,  
where the proportion of positive responses is typically low.  
This makes predictive modeling more challenging and requires targeted methods  
to improve performance.  
The research shows that **machine learning can effectively support marketing decision-making**,  
optimizing both **cost efficiency** and **targeting accuracy**.  

## Tech stack
- Python  
- scikit-learn  
- XGBoost  
- imbalanced-learn  
- pandas, numpy  
- matplotlib  

## Dataset
[Bank Marketing dataset – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
