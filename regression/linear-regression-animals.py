import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time

start_time = time.time()

df = pd.read_excel("/animals_data.xlsx")

X = df[["Weight_kg"]]
y = df["Speed_kmh"]

# Create and train linear regression model
model = LinearRegression()
model.fit(X, y)
# Predict speed for a given weight
def predict_speed(weight):
    input_df = pd.DataFrame([[weight]], columns=["Weight_kg"])
    predicted_speed = model.predict(input_df)[0]
    return round(predicted_speed, 2)

# Giving our target weight for predict
target_weight = 150


# Generating plot and its' values
x_range = pd.DataFrame({
    "Weight_kg": range(int(df["Weight_kg"].min()), int(df["Weight_kg"].max()) + 1)})
y_pred = model.predict(x_range)

plt.figure(figsize=(12, 7))

# Plot actual data points
plt.scatter(df["Weight_kg"], df["Speed_kmh"], color="blue", label="Actual Data")

# Annotate each point with the animal name
for i, row in df.iterrows():
    plt.text(row["Weight_kg"] + 10, row["Speed_kmh"], row["Animal"], fontsize=8)

# Plot the regression line
plt.plot(x_range, y_pred, color="red", linewidth=2, label="Regression Line")

# Predict and plot the speed for a specific weight (e.g., 1000 kg)
predicted_speed = predict_speed(target_weight)
plt.scatter([target_weight], [predicted_speed], color="red", s=100, zorder=5, label=f"Prediction for {target_weight} kg")

# Final plot adjustments
plt.xlabel("Weight (kg)")
plt.ylabel("Speed (km/h)")
plt.title("Animal Speed Prediction Based on Weight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




