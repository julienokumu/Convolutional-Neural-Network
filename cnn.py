import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader # load data in batches
import torchvision.transforms as transforms # preprocessing data
import torchvision # datasets
import matplotlib.pyplot as plt # image visualization
import os # file handling

# save predicted image with model's label
def save_predicted_image(image, predicted_label, number):
    image = image.squeeze().numpy() # convert tensor to numpy array
    plt.imshow(image, cmap="grey")
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off") # for better visualization
    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    file_path = f"predictions/prediction_{number}.png"
    plt.savefig(file_path)
    print(f"Prediction saved at: {file_path}")

# cnn model
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
        
        # second fully connected layer, output
        self.fc2 = nn.Linear(128, 10) # 10 neurons rep 0-9 digits

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x))) # apply first conv, relu and pool
        x = self.pool(self.relu2(self.conv2(x))) # apply second conv, relu and pool
        x = x.view(x.size(0), -1) # flatten tensor before feeding to nn
        x = self.relu3(self.fc1(x)) # apply first fc layer
        x = self.fc2(x) # output layer
        return x 


# load mnist dataset with transformations
transform = transforms.Compose([
    transforms.ToTensor(), # convert image to tensor
    transforms.Normalize((0.1307,), (0.3081,)) # normalize image with mean and sd
])

# download and load train and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# use gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize model to chosen device
model = CNN().to(device)

# loss function: crossentropyloass for classification tasks
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0 # initial running loss
    correct = 0 # track correct prediction
    total = 0 # total predictions
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) # move data to appr device
        optimizer.zero_grad() # clear previous gradients
        outputs = model(images) # forward pass, get predicted outputs from model
        loss = criterion(outputs, labels) # compute loss between predicted outputs and actual labels
        loss.backward() # apply backprop
        optimizer.step() # update model weights
        running_loss += loss.item() # accumulate loss
        _, predicted = torch.max(outputs, 1) # get predicted labels
        correct += (predicted == labels).sum().item() # count correct predictions
        total += labels.size(0) # count total predictions
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%") # print loss after each epoch

# save trained model to a file
torch.save(model.state_dict(), "tritoncnn.pth")
print("Model training complete and saved as tritoncnn.pth")

# load trained model for evaluation
model.load_state_dict(torch.load("tritoncnn.pth", map_location=device))
model.eval()

# take user input for testing
number = int(input("Enter a digit(0-9) to test: "))
indices = [i for i, (img, label) in enumerate(test_dataset) if label == number] # find indices matching images
if not indices:
    print("No images found for this digit")
else:
    index = indices[0] # take first occurence of the digit
    image, label = test_dataset[index] # retreive image and label form dataset
    image = image.unsqueeze(0).to(device) # add batch dimension and move to device

    with torch.no_grad(): # disable gradient computation during testing
        output = model(image) # get model prediction
        predicted_label = torch.argmax(output, 1).item() # get index with highest probability class

    save_predicted_image(image.cpu(), predicted_label, number) # save and display predicted image
    print(f"Actual Label: {label}, Predicted Label: {predicted_label}")

