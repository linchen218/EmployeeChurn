#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:41:34 2024

@author: linchen
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


#Import Data
data = pd.read_csv('/Users/linchen/Documents/DSCI 431 STATS/Employee-Attrition.csv')

print(data.head())

#Check for NAs
na_values = data.isna()  # or df.isnull()

if na_values.any().any():
    print("There are missing values in the DataFrame.")
else:
    print("The DataFrame has no missing values.")
    
#Exploratory Data Analysis
age_data = data['Age']

plt.figure(figsize=(8, 6))  
plt.hist(age_data, bins=10, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

working_years = data['TotalWorkingYears']

plt.figure(figsize=(8, 6))  
plt.hist(working_years, bins=10, color='skyblue', edgecolor='black')
plt.title('Total Working Years Distribution')
plt.xlabel('Total Working Years')
plt.ylabel('Frequency')
plt.show()

income_data = data['MonthlyIncome']

plt.figure(figsize=(8, 6))  
plt.hist(income_data, bins=10, color='skyblue', edgecolor='black')
plt.title('Monthly Income Distribution')
plt.xlabel('Monthly Income')
plt.ylabel('Frequency')
plt.show()

gender_data = data['Gender']

plt.figure(figsize=(8, 6))  
plt.hist(gender_data, bins=2, color='skyblue', edgecolor='black')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(age_data,income_data, c='blue', alpha=0.5)
plt.title('Scatter Plot of Age vs. Income')
plt.xlabel('Age')
plt.ylabel('Income')
plt.grid(True)  
plt.show()

average_age = data['Age'].mean()
print(average_age)

median_age = data['Age'].median()
print(median_age)

average_income = data['MonthlyIncome'].mean()
print(average_income)

median_income = data['MonthlyIncome'].median()
print(median_income)

correlation_matrix = data.corr(method='pearson')

correlation_mask = (correlation_matrix > 0.7) & (correlation_matrix < 1.0)  # Exclude self-correlations

highly_correlated_pairs = []

for col in correlation_mask.columns:
    correlated_vars = correlation_mask.index[correlation_mask[col]]
    for correlated_var in correlated_vars:
        highly_correlated_pairs.append((col, correlated_var))

for pair in highly_correlated_pairs:
    print(f"Highly correlated pair: {pair[0]} and {pair[1]} (Correlation: {correlation_matrix.loc[pair[0], pair[1]]})")

#Remove highly correlated pairs
columns_to_remove = ['JobLevel', 'PercentSalaryHike','TotalWorkingYears','YearsAtCompany','YearsWithCurrManager']
data = data.drop(columns=columns_to_remove)

correlation_matrix = data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Split data into testing and training set
X = data.drop('Attrition', axis=1)
y = data['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
log_reg_model = LogisticRegression()
tree_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

log_reg_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

log_reg_pred = log_reg_model.predict(X_test)
tree_pred = tree_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

# Accuracy
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
tree_accuracy = accuracy_score(y_test, tree_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Decision Tree Accuracy:", tree_accuracy)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)
