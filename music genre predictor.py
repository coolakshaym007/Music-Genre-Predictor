# Checking the accuracy of the model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('C:\\Users\\DELL\\Desktop\\New folder\\Music Genre Predictor\\music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

predictions = model.predict(x_test)

score = accuracy_score(y_test,predictions)
print(f'{int(score*100)}%')

#Saving the trained model for the future use
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x,y)

model = joblib.dump(model ,'music_recommender.joblib.joblib')

# Making the predictions using the saved model

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

model = joblib.load('C:\\Users\\DELL\\Desktop\\New folder\\Music Genre Predictor\\music_recommender.joblib.joblib')

# sample prediction set
prediction = model.predict([[21,1]])