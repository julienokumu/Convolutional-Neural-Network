import torch # nn builidng
import torch.nn as nn # nn architecture
import torch.optim as optim # optimizer for backprop
import torchvision # datasets and pretrained models
import torchvision.transforms as transforms # preprocessing data
from torch.utils.data import DataLoader # load data in batches

# define cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # first conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # second conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # first fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()

        # second fully connected layer
        self.fc2 = nn.Linear(128, 10)

    # forward prop
    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x))) # first convolution, relu and pooling
        x = self.pool(self.relu2(self.conv2(x))) # second convolution, relu and pooling
        x = x.view(x.size(0), -1) # flatten tensor before feeding to nn
        x = self.relu3(self.fc1(x)) # first fully connected layer + relu
        x = self.fc2(x) # second fully connected layer, output
        return x

# load mnist dataset with transformations
transform = transforms.Compose([
    transforms.ToTensor(), # converts images to tensors
    transforms.Normalize((0.1307,), (0.3081,)) # normalize dataset with mean and sd
])

# download and load training data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# download and load test data
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# create data loaders for batch precessing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # shuffle trianing data
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) # dont shuffle tets data

# gpu/cpu allocation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize the cnn model to chose device
model = CNN().to(device)

# define loss function
criterion = nn.CrossEntropyLoss() # classification tasks

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# number of training epochs
epochs = 5
for epoch in range(epochs): # loop through dataset
    model.train() # set model to training mode
    running_loss = 0.0 # initialize loss accumulator
    for images, labels in train_loader: # iterate over training data
        images, labels = images.to(device), labels.to(device) # move data to gpu/cpu

        optimizer.zero_grad() # zero the gradients before backprop
        outputs = model(images) # compute predicted output, forward prop
        loss = criterion(outputs, labels) # compute loss
        loss.backward() # backprop, compute gradients
        optimizer.step() # update model weights using optimizer

        running_loss += loss.item() # accumulate loss

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}") # print average loss for each epoch

# testing phase
model.eval() # set model to evaluation mode
correct = 0 # track number of correct peredictions
total = 0 # track total number of test samples

with torch.no_grad(): # disable gradient calculation during testing
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # move data to cpu/gpu
        outputs = model(images) # forward pass, get model predictions
        _, predicted = torch.max(outputs, 1) # get class with highest probability
        total += labels.size(0) # count total test samples
        correct += (predicted == labels).sum().item() # count correct prediction

print(f"Models Accuracy: {100 * correct / total:.2f}%")