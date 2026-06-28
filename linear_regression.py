import numpy as np
import matplotlib.pyplot as plt
X = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([2, 4, 5, 4, 5, 7])
mean_x = np.mean(X)
mean_y = np.mean(Y)
numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum((X - mean_x) ** 2)
m = numerator / denominator
b = mean_y - (m * mean_x)
print("Slope (m):", round(m, 2))
print("Intercept (b):", round(b, 2))
Y_pred = m * X + b
print("\nActual Values :", Y)
print("Predicted Values :", np.round(Y_pred, 2))
mse = np.mean((Y - Y_pred) ** 2)
print("\nMean Squared Error (MSE):", round(mse, 2))
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.title("Linear Regression from Scratch")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()