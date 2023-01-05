# Making the predictions using the saved model

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

model = joblib.load('C:\\Users\\DELL\\Desktop\\New folder\\Music Genre Predictor\\music_recommender.joblib')

# sample prediction set
prediction = model.predict([[21,1]])
