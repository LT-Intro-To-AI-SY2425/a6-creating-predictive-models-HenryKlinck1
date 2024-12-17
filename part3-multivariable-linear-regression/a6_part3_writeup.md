# Part 3 - Multivariable Linear Regression Writeup

After completing `a6_part3.py` answer the following questions

## Questions to answer

1. What is the R Squared coefficient for your model? What does that mean about the model in relation to your data?
My R-squared is 0.85 which means that the predictions were fairly accurate and that the model is a good fit for the data
2. Is your model accurate? Why or why not?
it is accurate to a point where the line stops giving accurate data
3. What does the model predict a 10-year-old car with 89000 miles is worth? What about a car that is 20 years old with 150000 miles?
Predicted price for a 10-year-old car with 89000 miles: 8983.563022658662
Predicted price for a 20-year-old car with 150000 miles: 1912.1776797375314
4. You may notice that some of your predicted results are negative. This is occurring when the value of age and the mileage of the car are very high. Why do you think this is happening?
because the model is using a slope for its line of prediction so once the numbers are large enough the line goes negative, impliying you need to pay somone to take your car.