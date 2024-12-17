import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
predicted_prices = model.predict(x_test)
print(f"Coefficients: {np.round(model.coef_, 2)}")
print(f"Intercept: {np.round(model.intercept_, 2)}")
print(f"R-squared: {np.round(model.score(x_test, y_test), 2)}")

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")
for actual, predicted in zip(y_test, model.predict(x_test)):
    print(f"Actual Price: {round(actual, 2)}, Predicted Price: {round(predicted, 2)}")

# Plotting the predicted vs actual prices
plt.scatter(y_test, predicted_prices)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect fit line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Predicting prices
new_data = np.array([[89000, 10], [150000, 20]])
predictions = model.predict(new_data)
print(f"Predicted price for a 10-year-old car with 89000 miles: {predictions[0]}")
print(f"Predicted price for a 20-year-old car with 150000 miles: {predictions[1]}")
