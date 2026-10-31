# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM

To write a Python program to predict the marks scored by a student using the Simple Linear Regression model.

## EQUIPMENTS REQUIRED

1. Hardware – PC
2. Anaconda – Python 3.x Installation / Jupyter Notebook

## ALGORITHM

1. Import the required Python libraries such as NumPy, Pandas, Matplotlib, and Scikit-learn.
2. Create a dataset containing the number of hours studied and the corresponding marks scored.
3. Separate the independent variable (Hours) and dependent variable (Marks).
4. Split the dataset into training and testing data.
5. Create a Simple Linear Regression model.
6. Train the model using the training data.
7. Predict the marks for the test data.
8. Calculate the slope and intercept of the regression line.
9. Evaluate the model using Mean Squared Error and R² Score.
10. Predict the marks for a new number of study hours.
11. Plot the actual data points and the regression line.

## PROGRAM

```python

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV dataset
data = pd.read_csv("student_scores.csv")

# Display the dataset
print("Dataset:")
print(data)

# Display the first five records
print("\nFirst five records:")
print(data.head())

# Independent variable
X = data[['Hours']]

# Dependent variable
y = data['Scores']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Simple Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict scores for the test data
y_pred = model.predict(X_test)

# Display the coefficient and intercept
print("\nCoefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Display actual and predicted values
print("\nActual Scores:")
print(y_test.values)

print("\nPredicted Scores:")
print(y_pred)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Predict score for a new student
hours = 9.25
predicted_score = model.predict([[hours]])

print(
    "\nPredicted score for",
    hours,
    "hours of study:",
    predicted_score[0]
)

# Plot the actual data points
plt.scatter(
    data['Hours'],
    data['Scores'],
    label='Actual Data'
)

# Plot the regression line
plt.plot(
    data['Hours'],
    model.predict(data[['Hours']]),
    label='Regression Line'
)

plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Simple Linear Regression - Hours vs Scores')
plt.legend()
plt.show()

```

## OUTPUT


<img width="534" height="730" alt="image" src="https://github.com/user-attachments/assets/8d388920-855d-4cae-a275-df60e3e0cbb1" />



## RESULT

Thus, the Python program to implement the Simple Linear Regression model for predicting the marks scored by a student was written and verified successfully.
