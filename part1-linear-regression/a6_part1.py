import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

x = x.reshape(-1, 1)

model = LinearRegression().fit(x, y)

# Get model coefficients and r-squared value
coef = round(float(model.coef_[0]), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

# Print out the linear equation and r-squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

x_predict = 43

# Predict the blood pressure
prediction = model.predict([[x_predict]])

# Print the prediction
print(f"Prediction when x is {x_predict}: {prediction[0]}")

# Create a plot
plt.figure(figsize=(6, 4))

plt.scatter(x, y, c="yellow")

plt.scatter(x_predict, prediction, c="red")

plt.xlabel("Age (years)")
plt.ylabel("Blood Pressure (mmHg)")
plt.title("Blood Pressure by Age")

plt.plot(x, coef * x + intercept, c="orange", label="Line of Best Fit")

plt.legend()
plt.show()
