'''Flight Booking Cancellation Prediction Using Decision Tree'''
# To build a machine learning model that predicts whether a booked flight will be canceled or not,
# based on various attributes like source, destination, airline, travel class, halt 

import pandas as pd

#---------------LOAD DATASET-------------------
df=pd.read_csv(r"E:\Desktop\ML\DECISION_TREE\flight_booking_data.csv")

#-------------REPLACE STRING VALUE---------------
# BY LABEL_ENCODER
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Source_new']=le.fit_transform(df['Source'])
df['Destination_new']=le.fit_transform(df['Destination'])
df['Airline_new']=le.fit_transform(df['Airline'])
df['TravelClass_new']=le.fit_transform(df['TravelClass'])
df['Halt_new']=le.fit_transform(df['Halt'])

df.drop(['Source','Destination','Airline','TravelClass','Halt'],axis='columns',inplace=True)

print(df)

#------FEATURES--------------
x=df.drop(['Cancellation'],axis='columns') # Independent
y=df['Cancellation'] # Dependent

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
print("TRAIN MODEL SCORE:", model.score(x_train,y_train)) # 0.99

# TEST PREDICTION 
y_test_pred=model.predict(x_test)
print("TEST MODEL SCORE:", model.score(x_test,y_test)) # 0.77

# MANUAL PREDICTION

print(model.predict([[2,3,1,1,2.0,0]])) # [1] Cancel


src=int(input("Enter Source(Bangalore-0, Chennai-1,Delhi-2,Kolkata-3,Mumbai-4 ) :"))
dest=int(input("Enter Destination(Bangalore-0, Chennai-1,Delhi-2,Kolkata-3,Mumbai-4) :"))
al=int(input("Enter Airline:(Vistara-0,Air India-1,GoAir-2,IndiGo-3,SpiceJet-4) :"))
tc=int(input("Enter TravelClass(Business-0, Economy-1) :"))
dur=float(input("Enter Duration  :"))
hlt=float(input("Enter Halt (Stop -0 , Non-stop 1)  :"))
result=model.predict([[src,dest,al,tc,dur,hlt]])
if(result==1):
    print("********Flight Cancelled*******")
else:
    print(".......Flight Not Cancel Enjoy your trip.....")