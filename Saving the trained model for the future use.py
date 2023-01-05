#Saving the trained model for the future use
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns = ['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x,y)

model = joblib.dump(model ,'music_recommender.joblib')
