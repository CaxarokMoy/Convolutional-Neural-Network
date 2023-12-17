import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Завантажте набір даних MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.MNIST("data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST("data", train=False, download=True, transform=transform)

# Створіть модель
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = torch.nn.Linear(20 * 4 * 4, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Створіть об'єкт моделі
model = Net()

# Встановіть оптимізатор
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Встановіть функцію втрат
criterion = torch.nn.CrossEntropyLoss()

# Завантажте дані за допомогою DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Тренуйте модель
for epoch in range(10):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Передайте дані в модель
        outputs = model(images)

        # Розрахуйте втрати
        loss = criterion(outputs, labels)

        # Оптимізуйте модель
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Виведіть прогрес
        if i % 100 == 0:
            print(f"Epoch: {epoch + 1}, Iteration: {i}, Loss: {loss.item()}")

# Перевірте модель на тестовому наборі даних
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")
