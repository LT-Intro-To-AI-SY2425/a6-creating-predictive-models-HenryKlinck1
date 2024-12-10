import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")

x = data["Age"].values
y = data["Blood Pressure"].values

x = x.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(xtrain, ytrain)

# Get the model's coefficients and R-squared value
coef = round(float(model.coef_[0]), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)

# Print out the linear equation and R-squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

'''
**********TEST THE MODEL**********
'''

predict = model.predict(xtest)

predict = np.around(predict, 2)

print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index]  
    predicted_y = predict[index]  
    x_coord = xtest[index]  
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
