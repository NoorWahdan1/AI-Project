import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

#Read CSV Data
data = pd.read_csv('Mobile_App_Success_Regression.csv')
#Modifying Installs Column to be valid Integrers instead of (5,000+) by replacing Commas and '+' by ''
data['Installs'] = data['Installs'].str.replace(',', '').str.replace('+', '').astype(np.int64)
#Modifying Price Column to be valid Float instead of (300$) by replacing '$' by ''
data['Price'] = data['Price'].str.replace('$', '').astype(float)
#Printing Data Description
print(data[['Rating', 'Installs', 'Reviews', 'Price']].describe())

#Plotting 3D Figure to Show (Reviews, Price and Rating) as X, Y, Z in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = data['Reviews']
y = data['Rating']
z = data['Price']
ax.scatter(x,y,z, c='red', marker='o')
ax.set_xlabel('Reviews');
ax.set_ylabel('Rating');
ax.set_zlabel('Price');
plt.show()

#Selecting feature columns

feature_cols = ['Reviews', 'Price'] #Here we selected Reviews and Price Only

#Setting  feature columns "Reviews and Price" as X and Target "Rating" as Y
X = data[feature_cols]
Y = data['Rating']

#Splitting X and Y into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=0)

#Creating Linear Regression Model and Fitting Train Data to the model then predict the Test Data
cls = linear_model.LinearRegression()
cls.fit(X_train,Y_train)
prediction = cls.predict(X_test)

#Printing the Co-efficient, Intercept and Mean Square Error
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y_test, prediction))

#Predict Rating based on Number of Reviews and Price:
Reviews=int(input('Enter number of Reviews: '))
Price=float(input('Enter Price: '))
x_test = pd.DataFrame([[Reviews, Price]], columns=['x','y'])
y_test = cls.predict(x_test)
print('Predicted Rating is ' + str(float(y_test[0])))