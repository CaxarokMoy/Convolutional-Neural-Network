import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# Завантаження та підготовка набору даних MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Завантаження даних за допомогою DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Визначення класу моделі
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Створення об'єкта моделі та визначення оптимізатора та планувальника
model = Net()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
criterion = nn.CrossEntropyLoss()

# Шлях для збереження та відновлення моделі та оптимізатора
save_path = "saved_model.pth"

# Перевірка наявності збереженого стану моделі та оптимізатора
if torch.cuda.is_available():
    try:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        print(f"Resuming training from epoch {epoch + 1}, best accuracy: {best_accuracy * 100:.2f}%")
    except FileNotFoundError:
        epoch = 0
        best_accuracy = 0.0
else:
    epoch = 0
    best_accuracy = 0.0

# Тренування моделі
for epoch in range(epoch, 15):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch: {epoch + 1}, Iteration: {i}, Loss: {loss.item()}")

    # Оцінка точності на тестовому наборі
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

    # Збереження моделі при кращій точності
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy
        }, save_path)
        print("Model saved successfully")

        # Застосування шагу планувальника швидкості навчання
        scheduler.step()

# Збереження остаточного стану моделі та оптимізатора
final_save_path = "final_model.pth"
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_accuracy': best_accuracy
}, final_save_path)
print(f"Final model state saved to {final_save_path}")

