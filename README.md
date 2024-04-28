# EmployeeChurn

## Introduction
The main objective of this project is to create a predictive model that will be able to predict employee churn as accurate as possible using various attributes associated with employees. 
Predicting employee churn offers invaluable advantages to organizations. It enables them to proactively implement retention strategies, addressing issues that may prompt employees to leave. 
This approach not only retains valuable talent but is also likely more cost-effective than recruiting and training new employees. 
To prepare for the data modeling process, I will go through the steps for data cleaning, data exploration, and data preparation in order to prepare the data for modeling. Various methods
and changes will occur throughout to adjust for better performance of the model. I will then go through the results of the model and evaluate the model’s performance to determine the success of 
the model as well as any next steps.

## Methods
* Data Cleaning
* EDA
* Predictive Modeling
  * Logistic Regression
  * Decision Tree
  * kNN
* Evaluation

To start off, the dataset will be cleaned to address any potential NAs before moving into the Exploratory Data Analysis step of the project. This step is crucial as it is important to understand what type of data we are looking at and the distribution of the variables and find hidden patterns. During this stage, we will also be preparing the data for modeling. I am sure that I will be going back and forth between modeling and this stage based on results or biases that I see in the results. For the modeling stage, I am currently considering a logistic regression, decision tree, and k-nearest neighbor and evaluating the results of the 3 models against each other. I will then go in depth into the performance metrics of the models and recommend the best model for the organization to utilize to predict or prepare for employee attrition/churn. 

Note: In order to run the code, please make sure to first download the dataset utilized in the code and save that in a location that is easily traceable. Please then change the path to load the data to reflect where it is saved in your computer before running the code.  

## Demo Application
This demo application is intended for an HR personnel at a company without any data science experience. The goal is to allow for individuals to easily manipulate the data they have in different ways that can help tell a story or identify any problems. This would also be useful to have for the exploratory data analysis portion before modeling occurs. It can help give the organization insight into any next steps that would be needed.

### How to use:
1. Download the demo code and replace the file path (notated in the code) with the path where the dataset is actually saved to ensure it runs
2. Run the code
3. Open up the terminal on your device and type in "cd" and your filepath where the code is saved
4. Open up a browser and copy in the url "http://127.0.0.1:8050"

### Next Steps:
Based on data analysis, it can provide helpful insight as to what actions the organization should do next. As for the demo app itself, a good second implementation would be for an ability for the user to simply type in the relevant attributes for an employee and have the output show what the predicted attrition is. 

