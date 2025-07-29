import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
x = np.linspace(-1, 1, 200)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(*x.shape)

x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Network WITHOUT activation function
class NetWithoutActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 20),
            nn.Linear(20, 20),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.model(x)

# Network WITH activation function (ReLU)
class NetWithActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, x, y, epochs=1000):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    return model

model_no_act = train_model(NetWithoutActivation(), x_tensor, y_tensor)
model_with_act = train_model(NetWithActivation(), x_tensor, y_tensor)

y_pred_no_act = model_no_act(x_tensor).detach().numpy()
y_pred_with_act = model_with_act(x_tensor).detach().numpy()

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Without Activation Function")
plt.scatter(x, y, label='True Data', color='gray', alpha=0.5)
plt.plot(x, y_pred_no_act, label='Predicted', color='red')
plt.legend()

plt.subplot(1, 2, 2)
plt.title("With Activation Function (ReLU)")
plt.scatter(x, y, label='True Data', color='gray', alpha=0.5)
plt.plot(x, y_pred_with_act, label='Predicted', color='green')
plt.legend()

plt.tight_layout()
plt.show()
