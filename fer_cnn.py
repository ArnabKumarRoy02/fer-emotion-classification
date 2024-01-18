import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import os

# WandB - Initialize a new run
wandb.init(project='facial-emotion-recognition',
           config={
               "epochs": 30,
               "batch_sizes": 64,
               "lr": random.uniform(1e-6, 1e-3),
               "data_augmentation": True,
           })

# Set the device to MPS
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


# Defining the CNN Model
class EmotionClassifierCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    # Make the forward pass
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
    

# Data transformations
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(17),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.ToTensor(),
])    

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Load the dataset
train_dataset = datasets.ImageFolder(root='fer-2013/train', transform=train_transform)
test_dataset = datasets.ImageFolder(root='fer-2013/test', transform=test_transform)

# WandB config
config = wandb.config
config.epochs = 30
config.batch_sizes = 64
config.lr = wandb.config.lr

print(f"\nFor Batch Size: {config.batch_sizes} with Learning Rate: {config.lr}")
# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_sizes, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_sizes, shuffle=False)

# Initialize the model
model = EmotionClassifierCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# Training loop
for epoch in range(config.epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass
        optimizer.zero_grad()   # Clear previously stored gradients
        outputs = model(inputs) # Make the forward pass
        loss = criterion(outputs, labels)   # Calculate the loss
        loss.backward() # Backward pass - Calculate the gradients
        optimizer.step()    # Update model parameters based on gradients and learning rate

    # Evaluation loop
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate validation loss and accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Log metrics to WandB
    wandb.log({"epoch": epoch + 1, "loss": loss.item(), "accuracy": accuracy})

    # Print the epoch, loss and accuracy
    print(f"Epoch: {epoch + 1}/{config.epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), 'emotion-classifier.pth')

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Log the confusion matrix
wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                   y_true=all_labels,
                                                   preds=all_preds,
                                                   class_names=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])})

# Save the confusion matrix
wandb.save("conf_mat.png")

# Finish the run
wandb.finish()