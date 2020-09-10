import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading data
data = pd.read_csv('tennis.csv')

# Transforming to numeric
from sklearn.preprocessing import LabelEncoder
le_data = data.apply(LabelEncoder().fit_transform)

# OHE 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
outlook = ohe.fit_transform(le_data.iloc[:, 0:1]).toarray()

wheather = pd.DataFrame(data=outlook, columns=['overcast','rainy','sunny'])
wheather = pd.concat([wheather,le_data.iloc[:,-2:]], axis=1)
wheather = pd.concat([wheather,data.iloc[:,1:3]], axis=1)


# Trained the wheather data
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(wheather.iloc[:,:-1],wheather.iloc[:,-1], test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

# P-value 
import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values= wheather.iloc[:,:-1], axis=1)
X_l = wheather.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog= wheather.iloc[:,-1:], exog=X_l)
r = r_ols.fit()
print(r.summary())

# Windy column' P value is 0.593 
# Calculated  the P values again without windy columns (Backward Elimination)

wheather2 = wheather.drop(['windy'], axis=1)

X = np.append(arr = np.ones((14,1)).astype(int), values= wheather.iloc[:,:-1], axis=1)
X_l = wheather2.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog= wheather2.iloc[:,-1:], exog=X_l)
r = r_ols.fit()
print(r.summary())

# In the end, no p value is greater than 0.5
# Values trained again
X_train , X_test, y_train, y_test = train_test_split(wheather2.iloc[:,:-1],wheather2.iloc[:,-1], test_size=0.33, random_state=0)
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)
y2_pred = lin_reg.predict(X_test)

