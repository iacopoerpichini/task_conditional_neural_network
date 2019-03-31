from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

kernel_size = 3



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=1)
        self.maxPool1 = nn.MaxPool2d(kernel_size=kernel_size, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1)
        self.maxPool2 = nn.MaxPool2d(kernel_size=kernel_size, stride=3)
        self.fc1 = nn.Linear(576, 256)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_drop = nn.Dropout(0.2)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.maxPool1(self.conv1(x)))
        x = F.relu(self.maxPool2(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc1(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        x = self.fc2_drop(x)
        return F.log_softmax(x, dim=1)

# fare un metodo della rete che sceglie iperparametri convoluzioni o definirli in cima alla classe

if __name__ == '__main__':
    model = Net()
    print(model)