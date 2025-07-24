#Understanding NN by implementign XOR gate

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_deriv(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

input_size = 2
hidden_size = 4
output_size = 1
lr = 0.1
epochs = 10000

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

for epoch in range(epochs):
    # FORWARD PROPAGATION
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # LOSS
    loss = binary_cross_entropy(y, a2)
    
    # BACKPROPAGATION
    dLoss_a2 = binary_cross_entropy_deriv(y, a2)
    d_a2_z2 = sigmoid_deriv(z2)
    d_z2_W2 = a1

    dLoss_z2 = dLoss_a2 * d_a2_z2
    dLoss_W2 = np.dot(d_z2_W2.T, dLoss_z2)
    dLoss_b2 = np.sum(dLoss_z2, axis=0, keepdims=True)

    d_z2_a1 = W2
    dLoss_a1 = np.dot(dLoss_z2, d_z2_a1.T)
    d_a1_z1 = relu_deriv(z1)
    dLoss_z1 = dLoss_a1 * d_a1_z1
    dLoss_W1 = np.dot(X.T, dLoss_z1)
    dLoss_b1 = np.sum(dLoss_z1, axis=0, keepdims=True)

    # UPDATE WEIGHTS
    W2 -= lr * dLoss_W2
    b2 -= lr * dLoss_b2
    W1 -= lr * dLoss_W1
    b1 -= lr * dLoss_b1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss: {loss:.4f}")

print("\nFinal output after training:")
print(a2.round())
