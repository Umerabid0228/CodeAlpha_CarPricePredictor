# CodeAlpha_CarPricePredictor

# Car Price Predictor

This project is a **Car Price Prediction Application** built using **Python, Tkinter**, and a **Random Forest Regressor** machine learning model. It allows users to input car details and get an estimated selling price in lakhs via a user-friendly GUI.

##  Features

* Graphical User Interface (GUI) built with **Tkinter** using a **Slate Blue** theme.
* Predicts car selling price based on:

  * Car Name (manual text input)
  * Year (Year the car was purchased)
  * Present Price (current ex-showroom price in lakhs)
  * Kms Driven (total distance driven)
  * Fuel Type (Petrol or Diesel)
  * Seller Type (Dealer or Individual)
  * Transmission Type (Manual or Automatic)
* **Predict** button for estimating price.
* **Reset** button to clear all inputs.
* Displays predicted price and model accuracy (R² score) in the interface.

##  Model & Data

* Model: **Random Forest Regressor** (with 100 estimators, random state 42)
* Dataset: `car data.csv` (used car dataset from Kaggle)
* Preprocessing:

  * Feature scaling using `StandardScaler`
  * One-hot encoding for categorical features
* Evaluation:

  * Train R²: **0.97**
  * Test R²: **0.96**
  * Train MSE: **0.81**
  * Test MSE: **0.84**
  * Cross-validation R² scores: \[0.98, 0.96, 0.95, 0.77, 0.96, 0.85, 0.97, 0.88, 0.94, 0.97]
  * Average CV R²: **0.92**

##  Feature Importances

Top features impacting prediction:

* Present\_Price: 0.8889
* Car\_Age: 0.0550
* Driven\_kms: 0.0311

##  Requirements

* Python 3.x
* Libraries:

  * `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tkinter`

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

##  How to Run

1. Place `car data.csv` in the same folder as the script.
2. Run the script.
3. Enter car details in the GUI and click **Predict**.

##  Notes

* Make sure numerical inputs are valid numbers.
* Car Name is manually typed (not a dropdown).
* Console output shows MSE, R² scores, CV scores, and feature importances.


