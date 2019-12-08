from joblib import load

key = 'k-NN'

best_sfs = load('models/best_sfs_scale.joblib')
knn = best_sfs.estimator
best_features = best_sfs.k_feature_names_

scaler = load('models/knn_scaler.joblib')
    
def pred(X):
    X = X.drop(columns=['lat', 'lon'])
    X[:] = scaler.transform(X)
    X_ant = X[list(best_features)]
    pred = knn.predict(X_ant)
    
    return pred