import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# data
alx_values = np.array([0.601, 0.506, 0.628, 0.650, 0.491, 0.551, 0.551, 0.583, 0.451, 0.537,
 0.459, 0.500, 0.495, 0.502, 0.419, 0.498, 0.507, 0.546, 0.520, 0.541,
 0.603, 0.547, 0.566, 0.586, 0.506, 0.592, 0.489, 0.485, 0.514, 0.557, 0.574])
sbp_values = np.array([111.68, 109.82, 113.03, 109.97, 115.38, 115.44, 115.77, 113.76, 114.68, 117.606,
 113.87, 115.32, 116.11, 114.30, 113.02, 110.91, 112.19, 119.51, 114.65, 113.43,
 116.93, 110.35, 114.41, 118.81, 112.87, 116.11, 116.95, 114.98, 116.39, 114.60, 118.30])
dbp_values = np.array([58.62, 59.17, 59.91, 56.24, 61.35, 62.01, 62.36, 62.428, 59.93, 62.48,
 59.58, 60.578, 62.18, 60.37, 59.01, 58.11, 59.32, 63.81, 61.01, 60.173,
 62.32, 57.69, 61.20, 62.12, 58.79, 61.22, 61.22, 60.84, 60.36, 59.92, 63.64])

X =alx_values.reshape(-1,1) # turns your 1-D ALX array into a column vector.

# sbp model 
model_sbp = LinearRegression().fit(X,sbp_values)
a_sbp     = model_sbp.coef_[0]
b_sbp     = model_sbp.intercept_
y_sbp_pred= model_sbp.predict(X)
r2_sbp    = r2_score(sbp_values, y_sbp_pred)

print(f"SBP  = {a_sbp:.4f} * ALX + {b_sbp:.4f}    (R² = {r2_sbp:.3f})")

# DBP model
model_dbp = LinearRegression().fit(X, dbp_values)
a_dbp     = model_dbp.coef_[0]
b_dbp     = model_dbp.intercept_
y_dbp_pred= model_dbp.predict(X)
r2_dbp    = r2_score(dbp_values, y_dbp_pred)

print(f"DBP  = {a_dbp:.4f} * ALX + {b_dbp:.4f}    (R² = {r2_dbp:.3f})")