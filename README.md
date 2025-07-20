# project_california_housing
This project predicts housing prices in California using a machine learning model trained on demographic and geographic data. It uses Random Forest Regressor for prediction and applies a clean, modular preprocessing pipeline built with scikit-learn.
The goal of this project is to build a regression model that can accurately predict house prices. The process involves data cleaning, preprocessing, training a machine learning model, and generating predictions on test data.

Key Features:

1.Stratified Sampling based on income categories to ensure representative train-test splits.

2.Preprocessing Pipeline using ColumnTransformer

3.Imputes missing values

4.Standardizes numeric data

5.One-hot encodes categorical data (ocean_proximity)

6.Random Forest Regression Model trained on processed data.

7.Model and Pipeline Saved as .pkl files using joblib for future use.

8.Prediction Output saved as output.csv with actual + predicted values.
