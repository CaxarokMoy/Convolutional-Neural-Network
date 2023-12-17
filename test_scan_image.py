import torch
import torchvision
import torchvision.transforms as transforms

# Завантажте набор даних MNIST
train_data = torchvision.datasets.MNIST(
    "data", train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST(
    "data", train=False, download=True, transform=transforms.ToTensor())

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
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, 20 * 4 * 4)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x



# Створіть об'єкт моделі
model = Net()

# Встановіть оптимізатор
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Встановіть функцію втрат
criterion = torch.nn.CrossEntropyLoss()

# Тренуйте модель
for epoch in range(10):
    for i, (images, labels) in enumerate(train_data):
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
            print(f"Епоха: {epoch}, Ітерація: {i}, Втрати: {loss.item()}")

# Перевірте модель на тестовому наборі даних
correct = 0
total = 0
for images, labels in test_data:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Точність: {correct / total}")
