import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time

start_time = time.time()

# Load dataset and use only the first 75 rows
df = pd.read_csv("student-mat.csv", sep=";").head(75)

# Define features and target
X = df[["G1", "G2"]]
y = df["G3"]

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict G3 values
y_pred = model.predict(X)

# Print predictions

plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='red', edgecolor='k', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'blue', linestyle='--', label='Ideal Prediction Line')
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Linear Regression: Predicting G3 from G1 and G2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nTotal execution time: {time.time() - start_time:.5f} seconds")
