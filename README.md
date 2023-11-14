# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
## Exp no:7
## Date:19.10.23
## AIM:

To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.

## Program:

```py
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SHARAN MJ
RegisterNumber: 212222240097
```

```py

import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

## data.head()

![Screenshot (237)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118344328/b82352d4-d32b-417f-b084-9bcc6679d6d4)

## data.info

![Screenshot (238)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118344328/f86626d8-e411-4d23-b827-c9be33f3f986)

## isnull() and sum()

![Screenshot (239)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118344328/9f764145-534a-4aaa-8573-d9a98b255e71)

## data.head() for salary

![Screenshot (240)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118344328/cb24a583-388f-48e2-a09a-a45e6128c12b)

## MSE value

![Screenshot (241)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118344328/ce29115f-52b7-417d-857e-1309409ddd1f)

## r2 value

![Screenshot (242)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118344328/7cf9ec95-7b09-4250-9a50-09b12bd94308)

## data prediction

![Screenshot (243)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118344328/cdd07dd7-6bfb-494f-9fe3-378d69126a31)

## Result:

Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
