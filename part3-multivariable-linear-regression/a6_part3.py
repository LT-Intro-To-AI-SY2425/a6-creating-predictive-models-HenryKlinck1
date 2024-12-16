import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values

#split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#create linear regression model
model = LinearRegression()
model.fit(x_train, y_train)
#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
print("***************")
print("Model Summary:")
print(f"Coefficients: {np.round(model.coef_, 2)}")
print(f"Intercept: {np.round(model.intercept_, 2)}")
print(f"R-squared: {np.round(model.score(x_test, y_test), 2)}")

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")
for actual, predicted in zip(y_test, model.predict(x_test)):
    print(f"Actual Price: {round(actual, 2)}, Predicted Price: {round(predicted, 2)}")