# SCRIPT THAT RUNS SEVERAL REGRESSION MODELS
#------------------------------------------------#

# import packages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# import the excel file
# to extract the file's path, right click and select copy pathname
# keep the csv file in the same directory as python script
file_path = 'ENTER FILE PATH AS STRING FOR CREDIT RISK CSV FILE'
df = pd.read_csv(file_path, low_memory = False)

# get total count of na
# checks whole dataframe
total_na = df.isna().sum().sum()
# check total share of NaN
na_share = round(total_na / len(df), 2)
# look at total NaN share by column
col_nan_share = (df.isna().sum() / len(df)).sort_values(ascending=False)
# drop loan_int_rate column since abt 10% is nan
df2 = df.drop(columns=['loan_int_rate'])
# drop NaN from new df
df2.dropna(inplace=True)
# check total % of columns dropped
total_drop_share = round(1 - len(df2) / len(df), 2)

# check the total share of defaults in dataset
# 1 is in default, 0 is not default
default_share = sum(df2['loan_status']) / len(df2)

# convert default on file to 1 or 0
df2['prior_known_default'] = (df2['cb_person_default_on_file'] == 'Y').astype(int)
# drop old column
df2.drop(columns=['cb_person_default_on_file'], inplace=True)

# create dummies for home ownership
encoded_home_ownership = pd.get_dummies(df2['person_home_ownership']).astype(int)
# concat back to df
df2 = pd.concat([df2, encoded_home_ownership], axis=1)
# drop the original column
df2.drop(columns=['person_home_ownership', 'OTHER'], inplace=True)

# extract the age column and see the max age
age_col = df['person_age'].sort_values(ascending=False).unique()
# eliminate ages above 100
ages_over_100 = age_col[age_col >= 100]
# drop the rows based on those values
df2 = df2[~df2['person_age'].isin(ages_over_100)]

# check for outliers in the person_emp_length column
emp_len_col = df2['person_emp_length'].sort_values(ascending=False)
# create list of outlier values
emp_len_over_100 = emp_len_col[emp_len_col >= 100]
df2 = df2[~df2['person_emp_length'].isin(emp_len_over_100)]

# check the emp len column to see if excessive amount of zeros
emp_unique_count = df2['person_emp_length'].value_counts()
# get % that are zero years work experience
percent_0_work_his = emp_unique_count.iloc[1] / len(df2)
# seems rather fishy that 12% have zero work experience
# look at rest of data where emp len is 0
no_emp_his_df = df2[df2['person_emp_length'] == 0]
# this column seems unreliable
# condraticary inputs that make me think this data is not useful
df2.drop(columns=['person_emp_length'], inplace=True)

# there are two columns that indicated the same thing
# income and loan to income ratio
# loan to income ratio makes more sense to keep
df2.drop(columns=['person_income'], inplace=True)

# create dummies for loan grade and loan intent
loan_intent_dummies = pd.get_dummies(df2['loan_intent'], prefix='intent').astype(int)
loan_grade_dummies = pd.get_dummies(df2['loan_grade'], prefix='loan_grade').astype(int)
# concat with original df
df3 = pd.concat([df2, loan_grade_dummies, loan_intent_dummies], axis=1)
# drop old columns
df3.drop(columns=['loan_intent', 'loan_grade'], inplace=True)

# there are a couple of columns where % loan income is 0 which makes no sense
# only a few rows with this incomplete data so we can just drop them
total_no_percent_income = sum((df3['loan_percent_income'] == 0).astype(int))
df3 = df3[df3['loan_percent_income'] != 0]

# SET UP LOGISTIC REGRESSION MODEL FOR ALL CASES
#----------------------------------------------#

# import packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# import color map for bar chart
from matplotlib import cm
# use cross validation on each model
from sklearn.model_selection import cross_val_score

# extract global variables
# X: feature matrix, y: target variable
X = df3.drop(columns=['loan_status'])
y = df3['loan_status']
# set up train and test vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# set up metric names to create plots
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
# set up list to store mean cv scores
cv_means = []
cv_mean_ticks = ['Base Model', 'L2 Penalty', 'L1 Penalty', 'Elastic Net Penalty']
# create list of feature names to analyze impact on model
ind_variables = list(X.columns)

# base case - no penalities
#------------------------#
# initialize and train the logistic regression model
base_model = LogisticRegression(penalty=None)
base_model.fit(X_train, y_train)
# make predictions on the test set
y_pred = base_model.predict(X_test)
# perform cross validation on model
cv_scores_base = cross_val_score(base_model, X_train, y_train, cv=5, scoring='accuracy')
cv_mean_base = cv_scores_base.mean()
# append to list
cv_means.append(cv_mean_base)
# calculate metrics
base_accuracy = accuracy_score(y_test, y_pred)
base_precision = precision_score(y_test, y_pred)
base_recall = recall_score(y_test, y_pred)
base_f1 = f1_score(y_test, y_pred)
# store the metrics in list
base_metrics = [base_accuracy, base_precision, base_recall, base_f1]
# base color map
color = cm.Blues([0.2, 0.4, 0.6, 0.8])
# create bar chart of metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics, base_metrics, color=color, edgecolor='black')
plt.xlabel('Evaluation Metric', fontdict={'fontsize': 14})
plt.ylabel('Metric Score', fontdict={'fontsize': 14})
plt.title('Base Model Evaluation Metrics', fontdict={'fontsize': 16})
plt.ylim(0,1)
plt.show()

# summarize the results
#---------------------#
# extract coefficients of models
coefficients_base = base_model.coef_[0]
# get the intercept of the model
intercept_base = base_model.intercept_[0]
# rank the coefficients
base_coef_scores = [(ind_var, np.exp(coef_val)) for ind_var, coef_val in zip(ind_variables, coefficients_base)]
# sort vals from largest to smallest
base_coef_scores.sort(key = lambda x: x[1])
# convert to dataframe
# you can convert list of tuples to two column df
base_results = pd.DataFrame(base_coef_scores, columns=['Variable', 'Odds Ratio'])

# add l2 penalty term to model
#----------------------------#
l2_model = LogisticRegression(penalty='l2', solver='liblinear')
l2_model.fit(X_train, y_train)
# make predictions on the test set
y_pred = l2_model.predict(X_test)
# perform cross validation on model
cv_scores_l2 = cross_val_score(l2_model, X_train, y_train, cv=5, scoring='accuracy')
cv_mean_l2 = cv_scores_l2.mean()
# append to lsit
cv_means.append(cv_mean_l2)
# calculate metrics
l2_accuracy = accuracy_score(y_test, y_pred)
l2_precision = precision_score(y_test, y_pred)
l2_recall = recall_score(y_test, y_pred)
l2_f1 = f1_score(y_test, y_pred)
# store metrics in a list
l2_metrics = [l2_accuracy, l2_precision, l2_recall, l2_f1]
# create color map
color = cm.Reds([0.2, 0.4, 0.6, 0.8])
# create bar chart of metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics, l2_metrics, color=color, edgecolor='black')
plt.xlabel('Evaluation Metric', fontdict={'fontsize': 14})
plt.ylabel('Metric Score', fontdict={'fontsize': 14})
plt.title('L2 Penalty Model Evaluation Metrics', fontdict={'fontsize': 16})
plt.ylim(0,1)
plt.show()

# summarize the results
#---------------------#
# extract coefficients of models
coefficients_l2 = l2_model.coef_[0]
# get the intercept of the model
intercept_l2 = l2_model.intercept_[0]
# rank the coefficients
l2_coef_scores = [(ind_var, np.exp(coef_val)) for ind_var, coef_val in zip(ind_variables, coefficients_l2)]
# sort vals from largest to smallest
l2_coef_scores.sort(key = lambda x: x[1])
# convert to dataframe
# you can convert list of tuples to two column df
l2_results = pd.DataFrame(l2_coef_scores, columns=['Variable', 'Odds Ratio'])

# add l1 penalty term to model
#----------------------------#
l1_model = LogisticRegression(penalty='l1', solver='liblinear')
l1_model.fit(X_train, y_train)
# make predictions on the test set
y_pred = l1_model.predict(X_test)
# perform cross validation on model
cv_scores_l1 = cross_val_score(l1_model, X_train, y_train, cv=5, scoring='accuracy')
cv_mean_l1 = cv_scores_l1.mean()
# append to list
cv_means.append(cv_mean_l1)
# calculate the metrics
l1_accuracy = accuracy_score(y_test, y_pred)
l1_precision = precision_score(y_test, y_pred)
l1_recall = recall_score(y_test, y_pred)
l1_f1 = f1_score(y_test, y_pred)
# store metrics in a list
l1_metrics = [l1_accuracy, l1_precision, l1_recall, l1_f1]
# create color map
color = cm.Purples([0.2, 0.4, 0.6, 0.8])
# create bar chart of metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics, l1_metrics, color=color, edgecolor='black')
plt.xlabel('Evaluation Metric', fontdict={'fontsize': 14})
plt.ylabel('Metric Score', fontdict={'fontsize': 14})
plt.title('L1 Penalty Model Evaluation Metrics', fontdict={'fontsize': 16})
plt.ylim(0,1)
plt.show()


# summarize the results
#---------------------#
# extract coefficients of models
coefficients_l1 = l1_model.coef_[0]
# get the intercept of the model
intercept_l1 = l1_model.intercept_[0]
# rank the coefficients
l1_coef_scores = [(ind_var, np.exp(coef_val)) for ind_var, coef_val in zip(ind_variables, coefficients_l1)]
# sort vals from largest to smallest
l1_coef_scores.sort(key = lambda x: x[1])
# convert to dataframe
# you can convert list of tuples to two column df
l1_results = pd.DataFrame(l1_coef_scores, columns=['Variable', 'Odds Ratio'])

# add elastic net penalty to model
#-------------------------------#
en_model = LogisticRegression(penalty='l1', l1_ratio=0.5, solver='liblinear')
en_model.fit(X_train, y_train)
# make predictions
y_pred = en_model.predict(X_test)
# perform cross validation on model
cv_scores_en = cross_val_score(en_model, X_train, y_train, cv=5, scoring='accuracy')
cv_mean_en = cv_scores_en.mean()
# append to list
cv_means.append(cv_mean_en)
# calculate the metrics
en_accuracy = accuracy_score(y_test, y_pred)
en_precision = precision_score(y_test, y_pred)
en_recall = recall_score(y_test, y_pred)
en_f1 = f1_score(y_test, y_pred)
# store metrics in a list
en_metrics = [en_accuracy, en_precision, en_recall, en_f1]
# create color map
color = cm.Oranges([0.2, 0.4, 0.6, 0.8])
# create bar chart of metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics, en_metrics, color=color, edgecolor='black')
plt.xlabel('Evaluation Metric', fontdict={'fontsize': 14})
plt.ylabel('Metric Score', fontdict={'fontsize': 14})
plt.title('Elastic Net Penalty Model Evaluation Metrics', fontdict={'fontsize': 16})
plt.ylim(0,1)
plt.show()

# summarize the results
#---------------------#
# extract coefficients of models
coefficients_en = en_model.coef_[0]
# get the intercept of the model
intercept_en = en_model.intercept_[0]
# rank the coefficients
en_coef_scores = [(ind_var, np.exp(coef_val)) for ind_var, coef_val in zip(ind_variables, coefficients_en)]
# sort vals from largest to smallest
en_coef_scores.sort(key = lambda x: x[1])
# convert to dataframe
# you can convert list of tuples to two column df
en_results = pd.DataFrame(en_coef_scores, columns=['Variable', 'Odds Ratio'])

# create bar chart of cv_means
#----------------------------#
# create color map
color = cm.Greens([0.2, 0.4, 0.6, 0.8])
# set up bar chart
plt.figure(figsize=(10, 6))
plt.bar(cv_mean_ticks, cv_means, color=color, edgecolor='black')
plt.xlabel('Model Penalty', fontdict={'fontsize': 14})
plt.ylabel('CV Mean Score', fontdict={'fontsize': 14})
plt.title('CV Mean Accuracy Score Evaluation by Penalty', fontdict={'fontsize': 16})
plt.ylim(0,1)
plt.show()

# export the data to excel
#------------------------#
# create the path and create writer
path = 'ENTER PATH FOR EXPORT TO EXISTING EXCEL FILE'
writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
# add results from each model
base_results.to_excel(writer, sheet_name='base-model', index=False)
l2_results.to_excel(writer, sheet_name='l2-results', index=False)
l1_results.to_excel(writer, sheet_name='l1-model', index=False)
en_results.to_excel(writer, sheet_name='en-model', index=False)

# it seems the most important variable by far is loan to income ratio
# we then separate into bins
df3['loan_income_bins'] = pd.cut(df3['loan_percent_income'],
                                 bins=[0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                                 labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%'])

# group by bins for further analysis
# find mean to see what percentage is in default by that group
default_rate_bins_means = df3.groupby('loan_income_bins')['loan_status'].mean()
default_rate_bins_count = df3.groupby('loan_income_bins')['loan_status'].sum()
# calculate share of total defaults by loan to income
total_defaults = np.sum(df3['loan_status'])
default_share = [defaults / total_defaults for defaults in default_rate_bins_count]
# create dictionary of all the results
loan_income_results = pd.DataFrame({'default_count': default_rate_bins_count, 'default_share': default_share, 'default_probabilty': default_rate_bins_means})
# export to summary excel file
loan_income_results.to_excel(writer, sheet_name='loan-income-ratio-res')
writer.close()

