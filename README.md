# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5. Determine training and test data set.
6. Apply decision tree regression on to the dataframe.
7. Get the values of Mean square error, r2 and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SHARAN MJ
RegisterNumber:  212222240097
*/

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
y = data["Salary"]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### data.head():
![240555705-3b090317-7639-4a3a-aef9-f89a18e73fa4](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393842/c55dbabf-c808-495d-8505-e54b6deaa67d)
### data.info():
![240556079-63f6ca58-2307-42ac-b695-796aae4d9e4b](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393842/6ec58b22-88d4-4b67-b041-1b64a453e427)
### isnull() and sum():
![240556172-cb64eb02-102a-4895-89d6-2b0afc6f5197](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393842/45993f14-45ed-4235-946e-fe5100ede328)
### data.head() for salary:
![240556248-3b0dae9a-6fef-4124-ac42-db6c1a33c4fa](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393842/22e7daee-2610-4409-a7db-6e694c4a3d9e)
### MSR value:
![240556281-364ae628-d50d-4984-b7c7-3a89edcc355f](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393842/e6154c70-402d-472e-a6ce-242cd6217056)
### r2 value:
![240556355-a1ad7b20-03c3-44f6-8234-fe52294a0534](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393842/bf065d7c-b523-4611-ba43-da754a4fb790)
### Data Prediction:
![240556487-7d9b4df8-6e94-40be-af97-43c59c9db7b7](https://github.com/srikarthickeyanganapathy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393842/e54265f9-4c22-4599-bca2-8f69b50987a5)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
