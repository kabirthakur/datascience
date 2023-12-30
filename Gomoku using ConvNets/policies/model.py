import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuCNN(nn.Module):
    def __init__(self):
        super(GomokuCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # New conv layer

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 15 * 15, 256)  # Adjusted for new conv layer
        self.fc2 = nn.Linear(256, 128)  # New fc layer
        self.fc3 = nn.Linear(128, 1)  # Output is a single heuristic value

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # Pass through the new conv layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Pass through the new fc layer
        x = self.fc3(x)  # Final output
        return x.squeeze()
