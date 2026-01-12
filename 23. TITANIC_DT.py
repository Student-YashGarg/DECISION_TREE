# TITANIC 

import pandas as pd 

df=pd.read_csv(r"E:\Desktop\ML\DECISION_TREE\titanic.csv")
print(df)

df.replace({'Sex':{'male':1,'female':0}},inplace=True)
print(df)
print(df.shape) # (891, 5)
print(df.isnull().sum()) # 177 null in age

# Drop null value
df.dropna(inplace=True)
print(df.shape) # (714, 5)

#--------FEATURES-----------
x=df.drop(['Survived'],axis='columns') # Independent
y=df['Survived'] # Dependent

#----SPLIT DATA------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#------IMPORT MODEL----------------------
# by Decision Tree
from sklearn import tree
model=tree.DecisionTreeClassifier()

#---------TRAINED MODEL----------------------------------------------
model.fit(x_train,y_train)
print("Model is Trained Successfully!")

#--------------PREDICTION----------------------------------

# TRAIN PREDICTION
y_train_pred=model.predict(x_train)
print("Actual Train Value:")
print(y_train)
print("TRAIN model prediction:")
print(y_train_pred)

print("TRAIN MODEL SCORE:", model.score(x_train,y_train)) # 0.99

# TEST PREDICTION 
y_test_pred=model.predict(x_test)
print("Actual TEST Value:")
print(y_test)
print("TEST model prediction:")
print(y_test_pred)

print("TEST MODEL SCORE:", model.score(x_test,y_test)) # 0.77


#----Prediction----------
# pclass,sex,age,fare
print(model.predict([[1,0,38,72]])) # [1]...Survived 

