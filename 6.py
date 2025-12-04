import numpy as np
import matplotlib.pyplot as plt

def lwr(x, X, Y, tau):
    w = np.exp(-(X - x)**2 / (2*tau*tau))
    W = np.diag(w)
    Xb = np.c_[np.ones(len(X)), X]
    theta = np.linalg.pinv(Xb.T @ W @ Xb) @ (Xb.T @ W @ Y)
    return np.array([1, x]) @ theta

X = np.linspace(0, 10, 30)
Y = np.sin(X) + np.random.randn(30)*0.2

Xq = np.linspace(0, 10, 200)
tau = 0.5
Yp = [lwr(x, X, Y, tau) for x in Xq]

plt.scatter(X, Y)
plt.plot(Xq, Yp, 'r')
plt.title("Locally Weighted Regression")
plt.show()
