import torch

class ResBlock(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(c_out)
        self.bn2 = torch.nn.BatchNorm2d(c_out)
        self.relu = torch.nn.ReLU()

        self.c_in = c_in
        self.c_out = c_out

        if c_in != c_out:
            self.conv_skip = torch.nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.c_in != self.c_out:
            y = self.conv_skip(y)

        x += y
        x = self.relu(x)

        return x

class ResNet18(torch.nn.Module):
    def __init__(self, num_cls=200):
        super(ResNet18, self).__init__()
        self.conv7x7 = torch.nn.Conv2d(3, 64, stride=2, kernel_size=7, padding=3)
        self.maxp3x3 = torch.nn.MaxPool2d(stride=2, kernel_size=3, padding=1)

        self.resblocks = torch.nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),

            ResBlock(64, 128),
            ResBlock(128, 128),

            ResBlock(128, 256),
            ResBlock(256, 256),

            ResBlock(256, 512),
            ResBlock(512, 512),
        )

        self.fc = torch.nn.Linear(512, num_cls)

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.maxp3x3(x)
        x = self.resblocks(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)

        return x

class ResNet34(torch.nn.Module):
    def __init__(self, num_cls=200):
        super(ResNet34, self).__init__()
        self.conv7x7 = torch.nn.Conv2d(3, 64, stride=2, kernel_size=7, padding=3)
        self.maxp3x3 = torch.nn.MaxPool2d(stride=2, kernel_size=3, padding=1)

        self.resblocks = torch.nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),

            ResBlock(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),

            ResBlock(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),

            ResBlock(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )

        self.fc = torch.nn.Linear(512, num_cls)

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.maxp3x3(x)
        x = self.resblocks(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)

        return x

class ResBlock_Bottle(torch.nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(c_in, c_mid, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(c_mid, c_out, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(c_mid)
        self.bn2 = torch.nn.BatchNorm2d(c_mid)
        self.bn2 = torch.nn.BatchNorm2d(c_out)
        self.relu = torch.nn.ReLU()

        self.c_in = c_in
        self.c_out = c_out

        if c_in != c_out:
            self.conv_skip = torch.nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.c_in != self.c_out:
            y = self.conv_skip(y)

        x += y
        x = self.relu(x)

        return x

class ResNet50(torch.nn.Module):
    def __init__(self, num_cls=200):
        super(ResNet50, self).__init__()
        self.conv7x7 = torch.nn.Conv2d(3, 64, stride=2, kernel_size=7, padding=3)
        self.maxp3x3 = torch.nn.MaxPool2d(stride=2, kernel_size=3, padding=1)

        self.resblocks = torch.nn.Sequential(
            ResBlock_Bottle(64, 64, 256),
            ResBlock_Bottle(256, 64, 256),
            ResBlock_Bottle(256, 64, 256),

            ResBlock_Bottle(256, 128, 512),
            ResBlock_Bottle(512, 128, 512),
            ResBlock_Bottle(512, 128, 512),
            ResBlock_Bottle(512, 128, 512),

            ResBlock_Bottle(512, 256, 1024),
            ResBlock_Bottle(1024, 256, 1024),
            ResBlock_Bottle(1024, 256, 1024),
            ResBlock_Bottle(1024, 256, 1024),
            ResBlock_Bottle(1024, 256, 1024),
            ResBlock_Bottle(1024, 256, 1024),

            ResBlock_Bottle(1024, 512, 2048),
            ResBlock_Bottle(2048, 512, 2048),
            ResBlock_Bottle(2048, 512, 2048),

        )

        self.fc = torch.nn.Linear(512, num_cls)

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.maxp3x3(x)
        x = self.resblocks(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)

        return x