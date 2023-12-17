import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Згенеруємо синтетичні дані для прикладу
torch.manual_seed(42)
X = torch.rand((100, 1), requires_grad=True)
y = 2 * X + 1 + 0.1 * torch.randn((100, 1))

# Створимо просту лінійну модель
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# Встановимо оптимізатор та критерій
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Тренуємо модель та зберігаємо значення MSE
epochs = 100
mse_values = []

for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    mse_values.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Відображення графіка MSE
plt.plot(range(1, epochs + 1), mse_values, marker='o', linestyle='-', color='b')
plt.title('Mean Squared Error Over Time')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig('mse_plot.png')
