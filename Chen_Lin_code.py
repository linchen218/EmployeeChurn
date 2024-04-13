#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:41:34 2024

@author: linchen
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score


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

attrition_data = data['Attrition']

plt.figure(figsize=(8, 6))  
plt.hist(attrition_data, bins=2, color='skyblue', edgecolor='black')
plt.title('Attrition Distribution')
plt.xlabel('Attrition')
plt.ylabel('Frequency')
plt.show()

dept_attrition = data.groupby(['Department', 'Attrition']).size().unstack(fill_value=0).reset_index()
fig, ax = plt.subplots()
bar_width = 0.35
bar_positions_left = range(len(dept_attrition['Department']))
bar_positions_stayed = [pos + bar_width for pos in bar_positions_left]
ax.bar(bar_positions_left, dept_attrition['Yes'], width=bar_width, label='Yes')

# Bar chart for employees who stayed
ax.bar(bar_positions_stayed, dept_attrition['No'], width=bar_width, label='No')

ax.set_xlabel('Department')
ax.set_ylabel('Number of Employees')
ax.set_title('Employees Left vs. Stayed by Department')
ax.set_xticks([pos + bar_width / 2 for pos in bar_positions_left])
ax.set_xticklabels(dept_attrition['Department'])
ax.legend()

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

#Upsample minority target class
df_majority = data[data['Attrition'] == 'No']
df_minority = data[data['Attrition'] == 'Yes']

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=len(df_majority),
                                 random_state=42)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

print(df_upsampled['Attrition'].value_counts())

#Distribution after upsampling
attrition_data = df_upsampled['Attrition']

plt.figure(figsize=(8, 6))  
plt.hist(attrition_data, bins=2, color='skyblue', edgecolor='black')
plt.title('Attrition Distribution')
plt.xlabel('Attrition')
plt.ylabel('Frequency')
plt.show()

#######

df_upsampled['Attrition'] = df_upsampled['Attrition'].map({'Yes': 1, 'No': 0})

X = df_upsampled.drop(['Attrition', 'EmployeeNumber'], axis=1)

y = df_upsampled['Attrition']

X = pd.get_dummies(X)

# Feature Selection
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# Feature importance threhold
threshold = 0.020
selected_features = X.columns[importances > threshold]
selected_importances = importances[importances > threshold]

sorted_indices = selected_importances.argsort()[::-1]
sorted_features = selected_features[sorted_indices]
sorted_importances = selected_importances[sorted_indices]


plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Importance Score > 0.020)")
plt.bar(range(len(sorted_features)), sorted_importances, align="center")
plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
plt.tight_layout()
plt.show()

# Split data into train and test set
X_selected = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Models
log_reg_model = LogisticRegression(max_iter = 3000)
tree_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

log_reg_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

log_reg_pred = log_reg_model.predict(X_test)
tree_pred = tree_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

# Log Regression
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
print("Logistic Regression Accuracy:", log_reg_accuracy)
log_reg_precision = precision_score(y_test, log_reg_pred)
print("Logistic Regression Precision:", log_reg_precision)
log_reg_recall = recall_score(y_test, log_reg_pred)
print("Logistic Regression Recall:", log_reg_recall)
log_reg_f1 = f1_score(y_test, log_reg_pred)
print("Logistic Regression F1:", log_reg_f1)
log_reg_conf_matrix = confusion_matrix(y_test, log_reg_pred)
log_reg_false_positives = log_reg_conf_matrix[0, 1]
log_reg_false_negatives = log_reg_conf_matrix[1, 0]
print(log_reg_conf_matrix)

# Decision Tree
tree_accuracy = accuracy_score(y_test, tree_pred)
print("Decision Tree Accuracy:", tree_accuracy)
tree_precision = precision_score(y_test, tree_pred)
print("Decision Tree Precision:", tree_precision)
tree_recall = recall_score(y_test, tree_pred)
print("Decision Tree Recall:", tree_recall)
tree_f1 = f1_score(y_test, tree_pred)
print("Decision Tree F1:", tree_f1)
tree_conf_matrix = confusion_matrix(y_test, tree_pred)
tree_false_positives = tree_conf_matrix[0, 1]
tree_false_negatives = tree_conf_matrix[1, 0]
print(tree_conf_matrix)

# K-nearest neighbors
knn_accuracy = accuracy_score(y_test, knn_pred)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)
knn_precision = precision_score(y_test, knn_pred)
print("K-Nearest Neighbors Precision:", knn_precision)
knn_recall = recall_score(y_test, knn_pred)
print("DK-Nearest Neighbors Recall:", knn_recall)
knn_f1 = f1_score(y_test, knn_pred)
print("K-Nearest Neighbors F1:", knn_f1)
knn_conf_matrix = confusion_matrix(y_test, knn_pred)
print(knn_conf_matrix)


# 10 fold cross validation
log_reg_model = LogisticRegression(max_iter=3000)
tree_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

# Logistic Regression
log_reg_scores = cross_val_score(log_reg_model, X_selected, y, cv=10, scoring='accuracy')
log_reg_mean_accuracy = log_reg_scores.mean()
log_reg_mean_precision = cross_val_score(log_reg_model, X_selected, y, cv=10, scoring='precision').mean()
log_reg_mean_recall = cross_val_score(log_reg_model, X_selected, y, cv=10, scoring='recall').mean()
log_reg_mean_f1 = cross_val_score(log_reg_model, X_selected, y, cv=10, scoring='f1').mean()

# Decision Tree
tree_scores = cross_val_score(tree_model, X_selected, y, cv=10, scoring='accuracy')
tree_mean_accuracy = tree_scores.mean()
tree_mean_precision = cross_val_score(tree_model, X_selected, y, cv=10, scoring='precision').mean()
tree_mean_recall = cross_val_score(tree_model, X_selected, y, cv=10, scoring='recall').mean()
tree_mean_f1 = cross_val_score(tree_model, X_selected, y, cv=10, scoring='f1').mean()

# K-nearest neighbors
knn_scores = cross_val_score(knn_model, X_selected, y, cv=10, scoring='accuracy')
knn_mean_accuracy = knn_scores.mean()
knn_mean_precision = cross_val_score(knn_model, X_selected, y, cv=10, scoring='precision').mean()
knn_mean_recall = cross_val_score(knn_model, X_selected, y, cv=10, scoring='recall').mean()
knn_mean_f1 = cross_val_score(knn_model, X_selected, y, cv=10, scoring='f1').mean()

print("Logistic Regression Mean Accuracy:", log_reg_mean_accuracy)
print("Logistic Regression Mean Precision:", log_reg_mean_precision)
print("Logistic Regression Mean Recall:", log_reg_mean_recall)
print("Logistic Regression Mean F1:", log_reg_mean_f1)

print("Decision Tree Mean Accuracy:", tree_mean_accuracy)
print("Decision Tree Mean Precision:", tree_mean_precision)
print("Decision Tree Mean Recall:", tree_mean_recall)
print("Decision Tree Mean F1:", tree_mean_f1)

print("K-Nearest Neighbors Mean Accuracy:", knn_mean_accuracy)
print("K-Nearest Neighbors Mean Precision:", knn_mean_precision)
print("K-Nearest Neighbors Mean Recall:", knn_mean_recall)
print("K-Nearest Neighbors Mean F1:", knn_mean_f1)
