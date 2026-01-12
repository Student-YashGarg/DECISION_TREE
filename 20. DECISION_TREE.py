# DECISION TREE
# MODEL PREDICT SALARY > 100K OR NOT
# classifier problem....can solve by both DT or LOGISTIC

import pandas as pd

df=pd.read_csv("E:\Desktop\ML\DECISION_TREE\salaries.csv")
print(df)
print(df.company.value_counts())
# company
# NetFlix         6
# Microsoft       6
# Cipla pharma    4
print(df.job.value_counts())
# job
# business manager       6
# sales executive        5
# computer programmer    5
print(df.degree.value_counts())
# degree
# bachelors    8
# masters      8

#---------REPLACE STING WITH VALUE----------------
# 1. via replace
df.replace({'degree':{'bachelors':0,'masters':1}},inplace=True)

# 2. via map
# df['degree']=df['degree'].map({'bachelors':0,'masters':1})

# 3. via label Encoder.................creates new column with tranform values 
# from sklearn.preprocessing import LabelEncoder
# # new_deg=LabelEncoder()
# df['new_degree']=LabelEncoder().fit_transform(df['degree'])
# df['new_job']=LabelEncoder().fit_transform(df['job'])
# df['new_degree']=LabelEncoder().fit_transform(df['degree'])

# 4. via OHE
job_dummy=pd.get_dummies(df['job'],dtype=int,drop_first=True) # sales,comp...drop busness
company_dummy=pd.get_dummies(df['company'],dtype=int,drop_first=True) # netflix,microsoft...drop cipla

new_df=pd.concat([company_dummy,job_dummy,df],axis='columns')
new_df.drop(['company','job'],axis='columns',inplace=True)
print(new_df)

#------IMPORT MODEL----------------------
# by Decision Tree
from sklearn import tree
model=tree.DecisionTreeClassifier()

# # by logistic Regression
# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression()

#------FEATURES-----------------
x=new_df.drop('salary',axis='columns') # independent
y=new_df['salary'] # dependent

#----SPLIT DATA------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#---------TRAINED MODEL----------------
model.fit(x_train,y_train)
print("Model is Trained Successfully!")

#--------------PREDICTION----------------------------------

# COMPANY       # JOB           # DEGREE
# cipla=0,0     # comp=1,0      # bachlor=0
# ms=1,0        # sales=0,1     # master=1
# netfilx=0,1   # business=0,0

# TRAIN PREDICTION
y_train_pred=model.predict(x_train)
print("Actual Train Value:")
print(y_train)
print("TRAIN model prediction:")
print(y_train_pred)

print("TRAIN MODEL SCORE:", model.score(x_train,y_train)) # 0.1

# TEST PREDICTION 
y_test_pred=model.predict(x_test)
print("Actual TEST Value:")
print(y_test)
print("TEST model prediction:")
print(y_test_pred)

print("TEST MODEL SCORE:", model.score(x_test,y_test)) # 0.8

# confusion matrix for test
print("TEST CONFUSION MATRIX:")
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_test_pred)
print(cm)

# MANUAL PREDICTION 

# MS,BUSINESS,BACHLOR
print("prediction for MS,BUSINESS,BACHLOR ", model.predict([[1,0,0,0,0]])) #[1]

