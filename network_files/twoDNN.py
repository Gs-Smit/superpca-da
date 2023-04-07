import torch.nn as nn
import torch.nn.functional as F


class TwoDNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(TwoDNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(100, 200, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(200, 400, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(400, 300)
        self.fc2 = nn.Linear(300, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
