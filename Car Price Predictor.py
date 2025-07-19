import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the data
df = pd.read_csv("car data.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# --- Metrics on Training Set ---
y_pred_train = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred_train)
r2 = r2_score(y, y_pred_train)
print("=== Model Performance on Training Data ===")
print("R² Score:", round(r2, 2))
print("Mean Squared Error (MSE):", round(mse, 2))

# --- Cross-validation ---
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
print("\n=== Cross-validation ===")
print("CV R² scores:", np.round(cv_scores, 2))
print("Average CV R²:", round(cv_scores.mean(), 2))

# --- Feature Importance ---
if hasattr(model, "feature_importances_"):
    print("\n=== Top 10 Feature Importances ===")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.head(10))

# --- GUI Design ---
root = tk.Tk()
root.title("Car Price Predictor")
root.geometry("550x650")
root.configure(bg="slate blue")

tk.Label(root, text="Car Price Predictor", font=("Arial", 20, "bold"), bg="slate blue", fg="white").pack(pady=15)

form_frame = tk.Frame(root, bg="slate blue")
form_frame.pack(pady=10)

labels_entries = {}

fields = {
    "Car Name": "",
    "Year (Year car was purchased)": "",
    "Present Price (in lakhs)": "",
    "Kms Driven": ""
}

for idx, (label_text, _) in enumerate(fields.items()):
    label = tk.Label(form_frame, text=label_text, font=("Arial", 12), bg="slate blue", fg="white")
    label.grid(row=idx, column=0, sticky="w", pady=5)
    entry = tk.Entry(form_frame, font=("Arial", 12), width=30)
    entry.grid(row=idx, column=1, pady=5)
    labels_entries[label_text] = entry

fuel_type = ttk.Combobox(form_frame, values=["Petrol", "Diesel"], font=("Arial", 12), state="readonly", width=28)
seller_type = ttk.Combobox(form_frame, values=["Dealer", "Individual"], font=("Arial", 12), state="readonly", width=28)
transmission = ttk.Combobox(form_frame, values=["Manual", "Automatic"], font=("Arial", 12), state="readonly", width=28)

tk.Label(form_frame, text="Fuel Type", font=("Arial", 12), bg="slate blue", fg="white").grid(row=4, column=0, sticky="w", pady=5)
fuel_type.grid(row=4, column=1, pady=5)

tk.Label(form_frame, text="Seller Type", font=("Arial", 12), bg="slate blue", fg="white").grid(row=5, column=0, sticky="w", pady=5)
seller_type.grid(row=5, column=1, pady=5)

tk.Label(form_frame, text="Transmission", font=("Arial", 12), bg="slate blue", fg="white").grid(row=6, column=0, sticky="w", pady=5)
transmission.grid(row=6, column=1, pady=5)

def predict_price():
    try:
        year = int(labels_entries["Year (Year car was purchased)"].get())
        present_price = float(labels_entries["Present Price (in lakhs)"].get())
        kms_driven = int(labels_entries["Kms Driven"].get())
        car_name = labels_entries["Car Name"].get().strip()

        fuel = fuel_type.get()
        seller = seller_type.get()
        trans = transmission.get()

        data = {
            "Year": year,
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Fuel_Type_Diesel": 1 if fuel == "Diesel" else 0,
            "Fuel_Type_Petrol": 1 if fuel == "Petrol" else 0,
            "Seller_Type_Individual": 1 if seller == "Individual" else 0,
            "Transmission_Manual": 1 if trans == "Manual" else 0
        }

        for col in X.columns:
            if col not in data:
                data[col] = 0

        input_df = pd.DataFrame([data])[X.columns]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        result_label.config(text=f"Estimated Selling Price: ₹ {prediction:.2f} lakhs")

    except Exception as e:
        messagebox.showerror("Input Error", f"An error occurred:\n{str(e)}\nPlease ensure all values are valid.")

def reset_form():
    for entry in labels_entries.values():
        entry.delete(0, tk.END)
    fuel_type.set("")
    seller_type.set("")
    transmission.set("")
    result_label.config(text="")

btn_frame = tk.Frame(root, bg="slate blue")
btn_frame.pack(pady=20)

predict_btn = tk.Button(btn_frame, text="Predict", font=("Arial", 14, "bold"), bg="white", fg="slate blue", width=12, command=predict_price)
predict_btn.grid(row=0, column=0, padx=10)

reset_btn = tk.Button(btn_frame, text="Reset", font=("Arial", 14, "bold"), bg="white", fg="slate blue", width=12, command=reset_form)
reset_btn.grid(row=0, column=1, padx=10)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="slate blue", fg="white")
result_label.pack(pady=10)

# Accuracy display on UI
accuracy_label = tk.Label(root, text=f"Model Accuracy (R²): {r2:.2f}", font=("Arial", 12, "bold"), bg="slate blue", fg="white")
accuracy_label.pack(pady=5)

root.mainloop()
