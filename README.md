# Car Price Prediction using Random Forest Regression

This code demonstrates the use of Random Forest Regression to predict car prices based on the horsepower of the car. It uses the CarPrice_Assignment.csv dataset to train the model and then evaluates its performance on a test set.

## Description

The code reads the car price dataset from a CSV file and selects the 'horsepower' column as the feature and 'price' as the target variable. It then splits the data into training and testing sets using the train_test_split function. A Random Forest Regression model is trained on the training data and then used to make predictions on the test data. The accuracy of the model is evaluated using the R-squared score for both the training and testing sets.

Additionally, the code also produces several plots including:
- A scatter plot of actual vs. predicted prices
- A scatter plot of horsepower vs. residuals
- A histogram of residuals

These visualizations provide insights into the model's performance.
