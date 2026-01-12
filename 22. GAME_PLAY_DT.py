# GAME PLAY PREDICTION

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
df_tennis=pd.read_csv(r"E:\Desktop\ML\DECISION_TREE\tennis.csv")
print(df_tennis)
print(df_tennis.shape)
print(df_tennis.info())
print(df_tennis.isnull().sum())

# REPLACE VALUES
# Label Encoding 
le=LabelEncoder()
df_tennis['outlook_new']=le.fit_transform(df_tennis['outlook'])
df_tennis['temp_new']=le.fit_transform(df_tennis['temp'])
df_tennis['humidity_new']=le.fit_transform(df_tennis['humidity'])
df_tennis['windy_new']=le.fit_transform(df_tennis['windy'])
df_tennis['play_new']=le.fit_transform(df_tennis['play'])

df_tennis=df_tennis.drop(['outlook','temp','humidity','windy','play'],axis='columns')
print(df_tennis)

# Features
x=df_tennis.drop(['play_new'],axis='columns')
y=df_tennis['play_new']

# Model Predication
from sklearn import tree
model=tree.DecisionTreeClassifier()

# Train without split data
model.fit(x,y)
print("Model Trained Done!")

# prediction 
print(model.predict([[0,1,0,0]]))
print(model.score(x,y))

# Manual prediction 

out1=int(input("Enter out look Weather(overcast-0,rainy-1,sunny-2) :"))
temp1=int(input("Enter Temperature(cool-0,hot-1,mild-2):"))
hum1=int(input("Enter Humidity:(high-0,normal-1)"))
wend1=int(input("Enter windy or not(TRUE-1,FALSE-0) :"))
result=model.predict([[out1,temp1,hum1,wend1]])
if(result==1):
    print("********PLayer Played a Tennis Game*******")
else:
    print(".......Player does not Played a Tennis Game.....")