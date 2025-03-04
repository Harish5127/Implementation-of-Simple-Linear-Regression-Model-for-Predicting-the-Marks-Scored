# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: R.Harish
RegisterNumber: 212224230085
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
df.head()

![Screenshot 2025-03-04 092654](https://github.com/user-attachments/assets/ecddbc22-71ac-4cfa-bea3-334372d35790)

df.tail()

![Screenshot 2025-03-04 092709](https://github.com/user-attachments/assets/09db01a9-ce40-4e16-8079-1ed846f3a183)

Array value of X

![Screenshot 2025-03-04 092840](https://github.com/user-attachments/assets/1299a45e-c6a4-462d-a15e-d80d4121af4d)

Array value of Y

![Screenshot 2025-03-04 092914](https://github.com/user-attachments/assets/85671496-3716-403e-8962-2ae657a47263)

Values of Y prediction

![Screenshot 2025-03-04 093008](https://github.com/user-attachments/assets/2d1681bb-5f2f-4382-be56-de433cb10f38)

Array values of Y test

![Screenshot 2025-03-04 093045](https://github.com/user-attachments/assets/a5d7dd0a-c773-4b75-88f1-81282ee2ed09)

Training Set Graph

![Screenshot 2025-03-04 093729](https://github.com/user-attachments/assets/7bb360ae-6976-40aa-909a-46ff3fc89833)

Test Set Graph

![Screenshot 2025-03-04 093737](https://github.com/user-attachments/assets/1057249d-d154-4709-b782-e28294237c29)

Values of MSE, MAE and RMSE

![Screenshot 2025-03-04 093742](https://github.com/user-attachments/assets/17cf7468-599c-4239-aaca-a9228c83e224)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
