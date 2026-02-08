import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time

start_time = time.time()

# Load dataset
df = pd.read_csv("/student-mat.csv", sep=";")
df = df.head(75)

# Selected features and target
features = ["sex", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "studytime", "internet", "G1", "G2"]
target = "G3"
data = df[features + [target]].copy()

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data[features]
y = data[target]

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction function
def predict_grade(sex, Pstatus, Medu, Fedu, Mjob, Fjob, studytime, internet, G1, G2):
    input_dict = {
        "sex": label_encoders["sex"].transform([sex])[0],
        "Pstatus": label_encoders["Pstatus"].transform([Pstatus])[0],
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": label_encoders["Mjob"].transform([Mjob])[0],
        "Fjob": label_encoders["Fjob"].transform([Fjob])[0],
        "studytime": studytime,
        "internet": label_encoders["internet"].transform([internet])[0],
        "G1": G1,
        "G2": G2
    }
    input_df = pd.DataFrame([input_dict])
    return round(model.predict(input_df)[0], 2)

# Example prediction
predicted = predict_grade("M", "T", 4, 4, "services", "teacher", 3, "yes", 16, 17)
print(f" Predicted final grade (G3): {predicted}")
print(f"------------------------------\n Executing time = {time.time() - start_time:.5f}")
