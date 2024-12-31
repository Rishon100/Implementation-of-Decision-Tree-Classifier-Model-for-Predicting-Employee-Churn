# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: V RISHON ANAND
RegisterNumber:  24900460
*/
```
```.py
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree 
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
![decision tree classifier model](sam.png)
![Screenshot 2024-11-25 084714](https://github.com/user-attachments/assets/76e23e1e-4dd0-4270-b44e-c679f5664e61)
![Screenshot 2024-11-25 085229](https://github.com/user-attachments/assets/3316fdb8-271b-46dd-9568-8392c58d1820)
![Screenshot 2024-11-25 084742](https://github.com/user-attachments/assets/8957b203-61e2-4ceb-bd08-c2db82d82cc4)
![Screenshot 2024-11-25 084748](https://github.com/user-attachments/assets/0e8e2057-13c2-4308-9304-a8f502f98410)
![Screenshot 2024-11-25 084754](https://github.com/user-attachments/assets/0e607f23-ff39-4279-b42d-556f6b37f02d)
![Screenshot 2024-11-25 084809](https://github.com/user-attachments/assets/e333cf60-d435-4b29-9c41-f956fccf75ff)
![Screenshot 2024-11-25 084821](https://github.com/user-attachments/assets/544913d2-a4cc-4b65-98da-faea875f85ed)
![Screenshot 2024-11-25 084831](https://github.com/user-attachments/assets/a183a5ad-b01e-42fd-b45b-91ef9222c1d0)
![Screenshot 2024-11-25 084838](https://github.com/user-attachments/assets/5455f0f8-8ca9-4385-bc73-593e82ba8583)
![Screenshot 2024-11-25 084843](https://github.com/user-attachments/assets/1a863ca3-881c-4c62-b4e1-98b9e6888f7e)
![Screenshot 2024-11-25 084916](https://github.com/user-attachments/assets/b15720f2-067a-4527-8578-7cba86aaa639)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
