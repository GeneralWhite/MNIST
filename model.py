from torch import nn

class FirstNet(nn.Module):
    def __init__(self):
        super(FirstNet, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(128*4*4, 64),
            nn.Dropout(),
            nn.Linear(64, 10)
        )

    def forward(self, output):
        output = self.module(output)
        return output