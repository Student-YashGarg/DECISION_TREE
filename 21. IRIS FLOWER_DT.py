# IRIS FLOWER CLASSIFICATION
# USING DECISION TREE...

import pandas as pd

#--------LOAD DATASET------------------------
df=pd.read_csv("E:\Desktop\ML\DECISION_TREE\Iris.csv")
print(df)
# # print(df.isnull().sum()) # 0
# print(df.shape) # (150, 6)

#--------REPLACE STRING VALUE on DEPENDENT (Y) COLUMN---------
# FOR DEPENDENT NEVER USE OHE...always use map/repale/labelEncoder
df.replace({'Species':{'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}},inplace=True)

# DROP ID ...IT NOT REQIUES
df.drop(['Id'],axis='columns',inplace=True)

print(df)
#------------FEATURES-------------------------------------------
x=df.drop(['Species'],axis='columns') # Independent
y=df['Species'] # Dependent

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

print("TRAIN MODEL SCORE:", model.score(x_train,y_train)) # 1.0

# TEST PREDICTION 
y_test_pred=model.predict(x_test)
print("Actual TEST Value:")
print(y_test)
print("TEST model prediction:")
print(y_test_pred)

print("TEST MODEL SCORE:", model.score(x_test,y_test)) # 0.97

# MANUAL PREDICTION
# results be like ...{'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

print(model.predict([[6.3,2.3,4.4,1.3]])) # [1].....Iris-versicolor

# spl=float(input("Enter value of SepalLength in Cm: "))
# spw=float(input("Enter value of SepalWidth in Cm: "))  
# ppl=float(input("Enter value of PetalLength in Cm: "))
# ppw=float(input("Enter value of PetalWidth in Cm: "))
# result=model.predict([[spl,spw,ppl,ppw]])
# if(result==0):
#     print("Type of Flower is Iris-setosa")
# elif(result==1):
#     print("Type of Flower is Iris-versicolor")
# else:
#     print("Type of Flower is Iris-virginica")

################################################################3
#--------GUI-------------------------------------
import tkinter as tk 
from tkinter import messagebox

app = tk.Tk()
app.title("IRIS FLOWER Detection")
app.geometry("500x500")
app.configure(bg="#f0f0f0")

# Entry labels and fields
fields = {
    "SepalLength (cm)": None,
    "SepalWidth (cm)": None,
    "PetalLength (cm)": None,
    "PetalWidth (cm)": None,
}

tk.Label(app, text="IRIS FLOWER DETECTION", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)
frame = tk.Frame(app, bg="#f0f0f0")
frame.pack()

for i, label in enumerate(fields):
    tk.Label(frame, text=label, font=("Arial", 12), bg="#f0f0f0").grid(row=i, column=0, pady=8, padx=10, sticky="w")
    entry = tk.Entry(frame, font=("Arial", 12), width=20)
    entry.grid(row=i, column=1, pady=8, padx=10)
    fields[label] = entry

# Prediction function
def predict_loan():
    try:
        sl = float(fields["SepalLength (cm)"].get())
        sw = float(fields["SepalWidth (cm)"].get())
        pl = float(fields["PetalLength (cm)"].get())
        pw = float(fields["PetalWidth (cm)"].get())
        features = [[sl,sw,pl,pw]]

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        # Above line calculates the probability that the loan will be approved.
        # Returns the probability of each class (0 or 1) as a list.
        # For example: [[0.10, 0.90]] 10% chance of Not Approved (class 0) 90% chance of Approved (class 1)

        label_map = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"}

        msg = f"Prediction:{label_map[prediction]}\n" \
              f"Login Probability: {prob*100:.2f}%\nModel Accuracy: {model.score(x_test,y_test)*100:.2f}%"
        messagebox.showinfo("Result", msg)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numerical values.")

# Button
tk.Button(app, text="Predict", command=predict_loan,
          font=("Arial", 12), bg="#4caf50", fg="white", padx=10, pady=5).pack(pady=20)

app.mainloop()



