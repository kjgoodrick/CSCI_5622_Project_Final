import pickle

import pandas as pd


CATEGORICAL_FEATURES = {
    'upper_mantle_vel_structure': range(1, 13),
}


key = 'GBRT'
with open('models/gbrt.pickle', 'rb') as _f:
    model = pickle.load(_f)


def _load_data_set(path=None, data=None):
    """
    Handle categorical variables as unordered.
    """
    if data is None:
        data = pd.read_csv(path)
    for categorical, categories in CATEGORICAL_FEATURES.items():
        unknowd_categories = set(data[categorical]).difference(categories)
        if unknowd_categories:
            msg = (f'Categorical feature {categorical} in {path} has unexpected value(s):\n'
                   + ', '.join(str(x) for x in unknowd_categories))
            raise ValueError(msg)
        dtype = pd.CategoricalDtype(categories)
        data[categorical] = data[categorical].astype(dtype)
    data = pd.get_dummies(data, columns=CATEGORICAL_FEATURES.keys())
    data = data[sorted(list(data), key=lambda s: s.lower())]

    return data

def pred(X):
    X_test = _load_data_set(data=X)
    for key in ['lon', 'lat', 'ghf']:
        if key in X_test:
            X_test.drop(key, axis=1, inplace=True)

    pred = model.predict(X_test)

    return pred
