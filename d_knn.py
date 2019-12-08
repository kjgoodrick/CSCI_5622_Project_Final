from joblib import load

key = 'd_k-NN'

knn = load('models/d_knn.joblib')
scaler = load('models/d_knn_scaler.joblib')
best_features = load('models/d_knn_labels.joblib')
    
def pred(X):
    X = X.drop(columns=['lat', 'lon'])
    X[:] = scaler.transform(X)
    X = X[list(best_features)]
    pred = knn.predict(X)
    
    return pred