# Corey Lang
# Coding provided by Corey Lang for this project and presentation
#
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error


df_reg = pd.read_csv("Data_Science_Fields_Salary_Categorization.csv", sep = ',')

df_reg.head()

sns.histplot(df_reg['Experience'], color='orange', edgecolor='brown')
plt.title('Experience')
plt.show()

sns.histplot(df_reg['Employment_Status'], color='orange', edgecolor='brown')
plt.title('Employment_Status')
plt.show()

sns.histplot(df_reg['Company_Location'], color='orange', edgecolor='brown')
plt.title('Experience')
plt.xticks(rotation=90)
plt.show()

sns.histplot(df_reg['Company_Size'], color='orange', edgecolor='brown')
plt.title('Company_Size')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.show()

sns.histplot(df_reg['Remote_Working_Ratio'], color='orange', edgecolor='brown')
plt.title('Remote_Working_Ratio')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
ticks = [0, 50, 100]
labels = ['0%', '50%', '100%']
plt.xticks(ticks=ticks, labels=labels)
plt.show()

exchange_rate = 0.013  # 1 INR = 0.013 USD as of 30th March 2023
df_reg['Salary_In_USD'] = df_reg['Salary_In_Rupees'].apply(lambda x: str(x).replace(',', '')).astype(float) * exchange_rate

df_reg['Salary_In_Rupees'] = df_reg['Salary_In_Rupees'].str[:-3]
def replace(str):
    string = str.replace(',','')
    return string
df_reg['Salary_In_Rupees_converted'] = df_reg.apply(lambda x: replace(x['Salary_In_Rupees']),axis=1)
df_reg['Salary_In_Rupees_converted'] = df_reg['Salary_In_Rupees_converted'].astype(int)
df_reg = df_reg.drop('Salary_In_Rupees', 1)
df_reg.rename(columns={'Unnamed: 0':'Count'}, inplace=True)

df_reg['Designation_average'] = df_reg.groupby(['Designation'])['Salary_In_Rupees_converted'].transform('mean')
df_reg['Employee_Location_average'] = df_reg.groupby(['Employee_Location'])['Salary_In_Rupees_converted'].transform('mean')


def group_designation(median):
    if median > 15000000:
        return 3
    elif median > 10000000 and median < 15000000:
        return 2
    elif median > 7000000 and median < 10000000:
        return 1
    else:
        return 0

df_reg['Designation_group'] = df_reg.apply(lambda x: group_designation(x['Designation_average']),axis=1)

def group_Employee_Location(median):
    if median > 10000000:
        return 3
    elif median > 7000000 and median < 10000000:
        return 2
    elif median > 5000000 and median < 7000000:
        return 1
    else:
        return 0

df_reg['Employee_Location_group'] = df_reg.apply(lambda x: group_Employee_Location(x['Employee_Location_average']),axis=1)

df_reg = df_reg.drop('Designation_average', 1)
df_reg = df_reg.drop('Employee_Location_average', 1)
df_reg = df_reg.drop('Count', 1)
df_reg = df_reg.drop('Designation', 1)
df_reg = df_reg.drop('Employee_Location', 1)
df_reg = df_reg.drop('Company_Location', 1)

Linear Regression Model

df_reg.columns

df_reg = df_reg[['Working_Year', 'Experience', 'Employment_Status', 'Company_Size',
       'Remote_Working_Ratio', 'Designation_group', 'Employee_Location_group',
        'Salary_In_Rupees_converted']]

X = df_reg.iloc[:, 0:7].values # feature variable
y = df_reg.iloc[:, 7].values # salary as target variable

S
label_encoder_Employment_Status = LabelEncoder()
label_encoder_Company_Size = LabelEncoder()

X[:,0] = label_encoder_Working_Year.fit_transform(X[:,0])
X[:,1] = label_encoder_experience.fit_transform(X[:,1])
X[:,2] = label_encoder_Employment_Status.fit_transform(X[:,2])
X[:,3] = label_encoder_Company_Size.fit_transform(X[:,3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)
y_standard = scaler.fit_transform(y.reshape(-1,1))

Srd, y_standard, test_size = 0.3, random_state = 1)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_normal_intercept = lr_model.intercept_
lr_normal_coef = lr_model.coef_
print('Coefficients: \n', lr_normal_coef)
lr_normal_score_train = lr_model.score(X_train, y_train)
print('The training set has a prediction accuracy of ', lr_normal_score_train)
lr_normal_score_test = lr_model.score(X_test, y_test)
print('The testing set has a predition accuracy of ', lr_normal_score_test)

# Make predictions using the testing set
ds_y_pred = lr_model.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, ds_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, ds_y_pred))

plt.scatter(y_test, ds_y_pred, color='orange', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='brown', linestyle='--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for Linear Regression Model')
plt.show()