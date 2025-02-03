
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms
import torchvision.datasets as datasets

import numpy as np
from torch.utils.data import Dataset

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LR = 1.0
EPOCHS = 14
GAMMA = 0.7
LOG_INTERVAL = 10

CUDA = "cuda"
device = torch.device(CUDA if torch.cuda.is_available() else "cpu")

class MNIST_Data(Dataset):
    def __init__(self, data_dir, label_dir):
        self.data = np.load(data_dir)
        self.label = np.load(label_dir)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx]).float()
        image = torch.reshape(image, (28, 28))
        label = self.label[idx]
        sample = (image, label)
        return sample

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: (img - torch.amin(img)) / (torch.amax(img) - torch.amin(img)))
        ])

    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = BATCH_SIZE,
        shuffle = True
        )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = TEST_BATCH_SIZE,
        shuffle = False
        )

    model = Classifier().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=LR)

    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)
    for epoch in range(1, EPOCHS + 1):
        #train(model, device, train_loader, optimizer, epoch)
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model, "classifier/classifier.pth")


if __name__ == '__main__':
    main()