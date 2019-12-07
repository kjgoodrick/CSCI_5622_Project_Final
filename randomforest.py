import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

key = 'Random Forest'

rf = RandomForestRegressor(n_estimators=100,
                           max_features='auto',
                           random_state=123456,
                           n_jobs=-1)

data_train = pd.read_csv('data/R17_global_train.csv')
X_train = data_train.drop(['lon', 'lat', 'GHF'], axis=1)
y_train = data_train[['GHF']].values.ravel()
rf.fit(X_train, y_train)

def pred(X):
    X = X.drop(['lon', 'lat'], axis=1)
    return rf.predict(X)