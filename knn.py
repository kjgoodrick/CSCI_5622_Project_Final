from joblib import load

key = 'knn'

best_sfs = load('models/best_sfs.joblib')
knn = best_sfs.estimator
best_features = best_sfs.k_feature_names_
    
def pred(X):
    X_test_selected = X[list(best_features)]
    pred = knn.predict(X_test_selected)
    
    return pred