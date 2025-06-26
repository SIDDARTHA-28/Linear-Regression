# Linear-Regression
Implement and understand simple and multiple linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plot style
plt.style.use('seaborn')

# Load and prepare the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# For simple linear regression: Predict petal length using petal width
X_simple = df[['petal width (cm)']]  # Feature
y = df['petal length (cm)']  # Target

# For multiple linear regression: Predict petal length using all other features
X_multiple = df[['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']]

# Split data into train and test sets (80-20 split)
X_simple_train, X_simple_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
X_multiple_train, X_multiple_test, _, _ = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

# Simple Linear Regression
simple_lr = LinearRegression()
simple_lr.fit(X_simple_train, y_train)
y_pred_simple = simple_lr.predict(X_simple_test)

# Evaluate Simple Linear Regression
mae_simple = mean_absolute_error(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print("Simple Linear Regression (Petal Width -> Petal Length):")
print(f"MAE: {mae_simple:.4f}")
print(f"MSE: {mse_simple:.4f}")
print(f"R-squared: {r2_simple:.4f}")
print(f"Coefficient: {simple_lr.coef_[0]:.4f}")
print(f"Intercept: {simple_lr.intercept_:.4f}")

# Multiple Linear Regression
multiple_lr = LinearRegression()
multiple_lr.fit(X_multiple_train, y_train)
y_pred_multiple = multiple_lr.predict(X_multiple_test)

# Evaluate Multiple Linear Regression
mae_multiple = mean_absolute_error(y_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

print("\nMultiple Linear Regression (All Features -> Petal Length):")
print(f"MAE: {mae_multiple:.4f}")
print(f"MSE: {mse_multiple:.4f}")
print(f"R-squared: {r2_multiple:.4f}")
print(f"Coefficients: {dict(zip(X_multiple.columns, multiple_lr.coef_))}")
print(f"Intercept: {multiple_lr.intercept_:.4f}")

# Plot Simple Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(X_simple_test, y_test, color='blue', label='Actual')
plt.plot(X_simple_test, y_pred_simple, color='red', label='Regression Line')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Simple Linear Regression: Petal Width vs Petal Length')
plt.legend()
plt.savefig('simple_linear_regression.png')
plt.close()

# Coefficient Interpretation
print("\nCoefficient Interpretation:")
print("Simple Linear Regression:")
print(f"- A 1 cm increase in petal width is associated with a {simple_lr.coef_[0]:.4f} cm increase in petal length.")
print("Multiple Linear Regression:")
for feature, coef in zip(X_multiple.columns, multiple_lr.coef_):
    print(f"- A 1 cm increase in {feature}, holding others constant, is associated with a {coef:.4f} cm change in petal length.")
