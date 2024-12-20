# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?
It went down to .69 or 69% because it is using inaccurate scales because the ranges of a salary are much larger than age range.
2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.
its .88 or 88% which is fairly good and I would argue is good enough because it is being used to predict if a car was bought or not.
3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?
I couldnt find a spacific trait that the model was commonly wrong about but with a larger test size I imigine their would be trends of incorectness maybe based on age or somthing.
4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.
No
