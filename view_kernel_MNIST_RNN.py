import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import csv

# Set a random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check for GPU availability and move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading and preparing MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize TensorBoard
writer = SummaryWriter()

# Define the neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 2000000)  # Changed the size of the output layer
        self.fc2 = nn.Linear(2000000, 10)  # Changed the size of the output layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Create model, optimizer, and learning rate scheduler
model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
criterion = nn.CrossEntropyLoss()

# Path for saving and restoring the model
save_path = "saved_model.pth"

# Check for the existence of a saved model
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

# Training the model
csv_path = "training_data.csv"
with open(csv_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Iteration', 'Loss', 'Accuracy'])

    for epoch in range(epoch, 15):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip gradients to avoid exploding gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch + 1}, Iteration: {i}, Loss: {loss.item():.4f}")

                # Log the loss to TensorBoard
                iteration = epoch * len(train_loader) + i
                writer.add_scalar('Loss/train', loss.item(), iteration)

                # Log the accuracy to TensorBoard
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
                print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")

                # Log the accuracy to TensorBoard
                writer.add_scalar('Accuracy/test', accuracy, iteration)

                # Save the training data to CSV
                csv_writer.writerow([epoch, i, loss.item(), accuracy])

        # Save the model if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }, save_path)
            print("Model saved successfully")

        # Learning rate scheduler step
        scheduler.step()

# Save the final model state
final_save_path = "final_model.pth"
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_accuracy': best_accuracy
}, final_save_path)

# Close TensorBoard writer
writer.close()

print(f"Final model state saved to {final_save_path}")
print(f"Training data saved to {csv_path}")
