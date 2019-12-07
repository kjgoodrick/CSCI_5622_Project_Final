import numpy as np
import pandas as pd
import matplotlib.pylab as plt
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

def plot_features():
    sort_i = np.argsort(rf.feature_importances_)
    imp = rf.feature_importances_
    imp = (imp-np.min(imp))/(np.max(imp) - np.min(imp))
    fig, ax = plt.subplots()
    plt.title('Feature Importance (Random Forest)')
    plt.barh(X_train.columns.values[sort_i], imp[sort_i])