import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Визначення класу моделі
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.fc4 = nn.Linear(2048, 10)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Завантаження та підготовка набору даних MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Завантаження даних за допомогою DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Створення об'єкта моделі та визначення оптимізатора та планувальника
model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=2, verbose=True)
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

# Ініціалізація графіку для точності
fig, ax = plt.subplots()
ax.set_title('Accuracy Over Time')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
plt.ion()

# Трекер точності
accuracy_values = []

# Трекер для early stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_accuracy):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0

# Тренування моделі
early_stopping = EarlyStopping(patience=5)

for epoch in range(epoch, 10):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
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
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        accuracy_values.append(accuracy)

        print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")

        # Оновлення графіка точності в реальному часі
        ax.plot(range(1, epoch + 2), accuracy_values, marker='o', linestyle='-', color='b')
        plt.pause(0.1)  # Пауза для оновлення графіка

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
            scheduler.step(best_accuracy)

            # Перевірка early stopping
            early_stopping(best_accuracy)
            if early_stopping.early_stop:
                print("Early stopping")
                break

# Збереження остаточного стану моделі та оптимізатора
final_save_path = "final_model.pth"
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_accuracy': best_accuracy
}, final_save_path)
print(f"Final model state saved to {final_save_path}")

# Завершення роботи з графіком після завершення тренування
plt.ioff()
plt.savefig('accuracy_plot_1mln_10_epoch.png')
