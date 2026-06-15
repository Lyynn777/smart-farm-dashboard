import pickle
model = pickle.load(open('model/yield_model.pkl', 'rb'))
print("Number of features:", model.n_features_in_)
print("Feature names:", model.feature_names_in_)