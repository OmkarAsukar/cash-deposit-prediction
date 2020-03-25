# --------------
# Import Libraries
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)

print(df.head())

df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ','_')

df = df.replace('NaN','np.nan')

df.isna().sum()
#print(df.head())
# Code ends here


# --------------
from sklearn.model_selection import train_test_split
df.set_index(keys='serial_number',inplace=True,drop=True)


# Code starts
df['established_date'] = pd.to_datetime(df['established_date'])
df['acquired_date'] = pd.to_datetime(df['acquired_date'])

X = df.drop('2016_deposits',axis = 1)
y = df['2016_deposits']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state = 3)


# Code ends here


# --------------
# time_col = X_train.select_dtypes(exclude=[np.number,'O']).columns
time_col = ['established_date', 'acquired_date']

# Code starts here
for col_name in time_col:
    new_col_name = "since_"+col_name
    X_train[new_col_name] = pd.datetime.now() - X_train[col_name]
    X_train[new_col_name] = X_train[new_col_name].apply(lambda x: float(x.days)/365)
    X_train.drop(columns=col_name,inplace=True)
    
    X_val[new_col_name] = pd.datetime.now() - X_val[col_name]
    X_val[new_col_name] = X_val[new_col_name].apply(lambda x: float(x.days)/365)
    X_val.drop(columns=col_name,inplace=True)


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
cat = X_train.select_dtypes(include='O').columns.tolist()

# Code starts here
X_train.fillna(0,inplace=True)
X_val.fillna(0,inplace=True)

le = LabelEncoder()
#for col in cat:
#X_train[col] = le.fit_transform(X_train[col])
#X_val[col] =  le.fit_transform(X_val[col])


X_train_temp = pd.get_dummies(data = X_train,columns = cat)
X_val_temp = pd.get_dummies(data =X_val, columns = cat)
# Code ends here


# --------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Code starts here
dt = DecisionTreeRegressor(random_state =5)
dt.fit(X_train,y_train)

accuracy = dt.score(X_val,y_val)

y_pred = dt.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val,y_pred))

print('Accuracy',accuracy)
print('rmse',rmse)




# --------------
from xgboost import XGBRegressor


# Code starts here
xgb = XGBRegressor(max_depth = 50,learning_rate = 0.83, n_estimators = 100)

xgb.fit(X_train,y_train)

accuracy = xgb.score(X_val,y_val)

y_pred = xgb.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val,y_pred))

print('Accuracy',accuracy)
print('rmse',rmse)



# Code ends here


