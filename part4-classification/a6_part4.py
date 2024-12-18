import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y
print(x)
print(y)

# Step 2: Standardize the data using StandardScaler
scaler = StandardScaler().fit(x)

# Step 3: Transform the data
x = scaler.transform(x)

# Step 4: Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Step 5: Create a LogisticRegression object and fit the data
model = linear_model.LogisticRegression()

# Step 6: Fit the model with training data
model.fit(x_train, y_train)

# Step 7: Print the score to see the accuracy of the model
print("Accuracy:", model.score(x_test, y_test))

# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
print("*************")
print("Testing Results:")
print("")

label_mapping = {0: "Not Purchased", 1: "Purchased"}

for index in range(len(x_test)):
    x_val = x_test[index]
    x_val = x_val.reshape(-1, 3) 
    y_pred = int(model.predict(x_val))

    actual = y_test[index]
    predicted_label = label_mapping[y_pred]
    actual_label = label_mapping[actual]

    print(f"Prediction: {predicted_label}, Actual: {actual_label}")
    print("")
