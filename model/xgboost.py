import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ----------- Step 1: Load Data -------------
npz_path = "/home/beatly-digital/Documents/indresh/indresh/alx/aggregated_metrics.npz"
data = np.load(npz_path)

alx = data["alx"]
hr = data["hr"]
sbp = data["sbp"]
dbp = data["dbp"]

X = np.column_stack((alx, hr))   # Input: ALX, HR
y = np.column_stack((sbp, dbp))  # Output: SBP, DBP


# ----------- Step 2: Split Data -------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# ----------- Step 3: XGBoost Regression -------------
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05,
                   max_depth=4, random_state=42)

model = MultiOutputRegressor(xgb)
model.fit(X_train, y_train)


# ----------- Step 4: Evaluation -------------
y_pred = model.predict(X_test)

mse_sbp = mean_squared_error(y_test[:, 0], y_pred[:, 0])
mse_dbp = mean_squared_error(y_test[:, 1], y_pred[:, 1])

mae_sbp = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
mae_dbp = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

r2_sbp = r2_score(y_test[:, 0], y_pred[:, 0])
r2_dbp = r2_score(y_test[:, 1], y_pred[:, 1])

print("\n--- Performance Metrics ---")
print(f"SBP  - MSE: {mse_sbp:.2f}, MAE: {mae_sbp:.2f}, R2: {r2_sbp:.3f}")
print(f"DBP  - MSE: {mse_dbp:.2f}, MAE: {mae_dbp:.2f}, R2: {r2_dbp:.3f}")


# ----------- Step 5: Visualization -------------
# SBP Plot
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
plt.plot([y_test[:, 0].min(), y_test[:, 0].max()],
         [y_test[:, 0].min(), y_test[:, 0].max()],
         'k--', lw=2)
plt.xlabel('True SBP')
plt.ylabel('Predicted SBP')
plt.title('SBP Prediction (ALX+HR -> SBP)')
plt.show()

# DBP Plot
plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
plt.plot([y_test[:, 1].min(), y_test[:, 1].max()],
         [y_test[:, 1].min(), y_test[:, 1].max()],
         'k--', lw=2)
plt.xlabel('True DBP')
plt.ylabel('Predicted DBP')
plt.title('DBP Prediction (ALX+HR -> DBP)')
plt.show()
