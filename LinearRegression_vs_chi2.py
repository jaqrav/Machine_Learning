import numpy as np
from matplotlib import pyplot as plt

# data
months = np.arange(1, 13)
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]
revenue = np.array(revenue)

# chi2 calculation
def chi2(x,y):
    Mx = np.sum(x)
    My = np.sum(y)
    Mxx = np.sum(x**2)
    Mxy = np.sum(x*y)
    N = len(x)
    D = N*Mxx - Mx**2
    a = (N*Mxy - Mx*My)/D
    b = (Mxx*My - Mxy*Mx)/D
    return a, b

a,b = chi2(months, revenue)
x_plot = np.linspace(min(months),max(months),50)
y_plot = a*x_plot + b

# Linear Regression - my method

# Gradient for a and b
def gradient_b(x, y, a, b):
    return -2*np.sum(y - a*x - b)/len(x)

def gradient_a(x, y, a, b):
    return -2*np.sum((y - a*x - b)*x)/len(x)

def update_gradient(x, y, a, b, learning_rate):
    grad_a = gradient_a(x, y, a, b)
    grad_b = gradient_b(x, y, a, b)
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    return a, b

def my_linear_regression(x, y, learning_rate, num_iterations):
    a = 0.0
    b = 0.0
    for _ in range(num_iterations):
        a, b = update_gradient(x, y, a, b, learning_rate)
    return a, b

def my_linear_regression_2(x, y, learning_rate, accurecy):
    a = 0.0
    b = 0.0
    diff = 1
    while diff > accurecy:
        a_old, b_old = a, b
        a, b = update_gradient(x, y, a, b, learning_rate)
        diff = max(abs(a - a_old), abs(b - b_old))
    return a, b

# Parameters for my linear regression
learning_rate = 0.01
num_iterations = 10000
a_my, b_my = my_linear_regression(months, revenue, learning_rate, num_iterations)
a_my_2, b_my_2 = my_linear_regression_2(months, revenue, learning_rate, 1e-6)
y_my_linear_regression = a_my * x_plot + b_my
y_my_linear_regression_2 = a_my_2 * x_plot + b_my_2

# Linea Regression - sklearn
months = months.reshape(-1,1)
from sklearn.linear_model import LinearRegression
fitter = LinearRegression()
fitter.fit(months, revenue)
a_sklearn = fitter.coef_[0]
b_sklearn = fitter.intercept_
y_sklearn = fitter.predict(x_plot.reshape(-1, 1))

print("chi2: a = {:.8}, b = {:.8}".format(a,b))
print("sklearn: a = {:.8}, b = {:.8}".format(a_sklearn, b_sklearn))
print("my LR: a = {:.8}, b = {:.8}".format(a_my, b_my))
print("my LR 2: a = {:.8}, b = {:.8}".format(a_my_2, b_my_2))

plt.figure()
plt.plot(months, revenue, 'ro', label = 'data')
plt.plot(x_plot, y_plot, 'b-', label = 'chi2')
plt.plot(x_plot, y_sklearn, 'g-', label = 'sklearn')
plt.plot(x_plot, y_my_linear_regression, 'y-', label = 'my LR')
plt.xlabel('months')
plt.ylabel('revenue')
plt.title("chi2: a = {:.4}, b = {:.4}".format(a,b) + "\nsklearn: a = {:.4}, b = {:.4}".format(a_sklearn, b_sklearn))
plt.legend()
plt.savefig('LR_vs_Chi2.png')
