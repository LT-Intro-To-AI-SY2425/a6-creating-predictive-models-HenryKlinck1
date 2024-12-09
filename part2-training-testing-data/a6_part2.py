import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

# Load the data from the CSV file
data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")

# Define the features (Age) and target variable (Blood Pressure)
x = data["Age"].values
y = data["Blood Pressure"].values

# Reshape x to be a 2D array for the model
x = x.reshape(-1, 1)

# Split the data into training and testing sets (80% train, 20% test)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Create the linear regression model and train it with the training data
model = LinearRegression().fit(xtrain, ytrain)

# Get the model's coefficients and R-squared value
coef = round(float(model.coef_[0]), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)

# Print out the linear equation and R-squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

# Predict the blood pressure for a person who is 43 years old
x_predict = 43
prediction = model.predict([[x_predict]])

# Print the prediction
print(f"Prediction when x is {x_predict}: {prediction[0]}")

'''
**********TEST THE MODEL**********
'''

# Get the predicted y values for the testing data
predict = model.predict(xtest)

# Round the predicted values to 2 decimal places
predict = np.around(predict, 2)

# Compare the actual and predicted values
print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index]  # Actual blood pressure value from ytest
    predicted_y = predict[index]  # Predicted blood pressure value
    x_coord = xtest[index]  # Age value from xtest
    print(f"x value: {float(x_coord[0])}, Predicted y value: {predicted_y}, Actual y value: {actual}")

'''
**********CREATE A VISUAL OF THE RESULTS**********
'''
plt.figure(figsize=(6, 4))

plt.scatter(xtrain, ytrain, c="yellow", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing Data")
plt.scatter(xtest, predict, c="red", label="Predictions")

plt.plot(x, coef * x + intercept, c="orange", label="Line of Best Fit")

plt.xlabel("Age (years)")
plt.ylabel("Blood Pressure (mmHg)")
plt.title("Blood Pressure by Age")

plt.legend()

plt.show()
