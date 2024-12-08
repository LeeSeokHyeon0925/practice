import torch

class conv_bn_relu(torch.nn.Module):
    def __init__(self, c_in, c_out, k_size=3, padding=1):
        super(conv_bn_relu, self).__init__()
        self.conv_bn_relu = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, kernel_size=k_size, padding=padding),
            torch.nn.BatchNorm2d(c_out),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_bn_relu(x)
        return x

class linear_relu_drop(torch.nn.Module):
    def __init__(self, c_in, c_out, dropout=0.2):
        super(linear_relu_drop, self).__init__()
        self.linear_relu_drop = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_out),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.linear_relu_drop(x)
        return x

class VGG11(torch.nn.Module):
    def __init__(self, num_classes=200):
        super(VGG11, self).__init__()
        # [b, c, h, w] // # [b, 3, 128, 128]
        self.conv = torch.nn.Sequential(
            conv_bn_relu(3, 64), # [b, 64, 128, 128]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 64, 64, 64]

            conv_bn_relu(64, 128), # [b, 128, 64, 64]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 128, 32, 32]

            conv_bn_relu(128, 256), # [b, 256, 32, 32]
            conv_bn_relu(256, 256), # [b, 256, 32, 32]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 256, 16, 16]

            conv_bn_relu(256, 512), # [b, 512, 16, 16]
            conv_bn_relu(512, 512), # [b, 512, 16, 16]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 512, 8, 8]

            conv_bn_relu(512, 512), # [b, 512, 8, 8]
            conv_bn_relu(512, 512), # [b, 512, 8, 8]
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # [b, 512, 4, 4]
        )

        self.fcs = torch.nn.Sequential(
            linear_relu_drop(512 * 4 * 4, 4096),
            linear_relu_drop(4096, 4096),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, (-1, 512 * 4 * 4))
        x = self.fcs(x)
        return x

class VGG13(torch.nn.Module):
    def __init__(self, num_classes=200):
        super(VGG13, self).__init__()
        # [b, c, h, w] // # [b, 3, 128, 128]
        self.conv = torch.nn.Sequential(
            conv_bn_relu(3, 64),  # [b, 64, 128, 128]
            conv_bn_relu(64, 64),  # [b, 64, 128, 128]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 64, 64, 64]

            conv_bn_relu(64, 128),  # [b, 128, 64, 64]
            conv_bn_relu(128, 128),  # [b, 128, 64, 64]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 128, 32, 32]

            conv_bn_relu(128, 256),  # [b, 256, 32, 32]
            conv_bn_relu(256, 256),  # [b, 256, 32, 32]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 256, 16, 16]

            conv_bn_relu(256, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 8, 8]

            conv_bn_relu(512, 512),  # [b, 512, 8, 8]
            conv_bn_relu(512, 512),  # [b, 512, 8, 8]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 4, 4]
        )

        self.fcs = torch.nn.Sequential(
            linear_relu_drop(512 * 4 * 4, 4096),
            linear_relu_drop(4096, 4096),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, (-1, 512 * 4 * 4))
        x = self.fcs(x)
        return x

class VGG16(torch.nn.Module):
    def __init__(self, num_classes=200):
        super(VGG16, self).__init__()
        # [b, c, h, w] // # [b, 3, 128, 128]
        self.conv = torch.nn.Sequential(
            conv_bn_relu(3, 64),  # [b, 64, 128, 128]
            conv_bn_relu(64, 64),  # [b, 64, 128, 128]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 64, 64, 64]

            conv_bn_relu(64, 128),  # [b, 128, 64, 64]
            conv_bn_relu(128, 128),  # [b, 128, 64, 64]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 128, 32, 32]

            conv_bn_relu(128, 256),  # [b, 256, 32, 32]
            conv_bn_relu(256, 256),  # [b, 256, 32, 32]
            conv_bn_relu(256, 256),  # [b, 256, 32, 32]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 256, 16, 16]

            conv_bn_relu(256, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 8, 8]

            conv_bn_relu(512, 512),  # [b, 512, 8, 8]
            conv_bn_relu(512, 512),  # [b, 512, 8, 8]
            conv_bn_relu(512, 512),  # [b, 512, 8, 8]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 4, 4]
        )

        self.fcs = torch.nn.Sequential(
            linear_relu_drop(512 * 4 * 4, 4096),
            linear_relu_drop(4096, 4096),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, (-1, 512 * 4 * 4))
        x = self.fcs(x)
        return x

class VGG19(torch.nn.Module):
    def __init__(self, num_classes=200):
        super(VGG19, self).__init__()
        # [b, c, h, w] // # [b, 3, 128, 128]
        self.conv = torch.nn.Sequential(
            conv_bn_relu(3, 64),  # [b, 64, 128, 128]
            conv_bn_relu(64, 64),  # [b, 64, 128, 128]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 64, 64, 64]

            conv_bn_relu(64, 128),  # [b, 128, 64, 64]
            conv_bn_relu(128, 128),  # [b, 128, 64, 64]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 128, 32, 32]

            conv_bn_relu(128, 256),  # [b, 256, 32, 32]
            conv_bn_relu(256, 256),  # [b, 256, 32, 32]
            conv_bn_relu(256, 256),  # [b, 256, 32, 32]
            conv_bn_relu(256, 256),  # [b, 256, 32, 32]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 256, 16, 16]

            conv_bn_relu(256, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 8, 8]

            conv_bn_relu(512, 512),  # [b, 512, 8, 8]
            conv_bn_relu(512, 512),  # [b, 512, 8, 8]
            conv_bn_relu(512, 512),  # [b, 512, 8, 8]
            conv_bn_relu(512, 512),  # [b, 512, 8, 8]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 4, 4]
        )

        self.fcs = torch.nn.Sequential(
            linear_relu_drop(512 * 4 * 4, 4096),
            linear_relu_drop(4096, 4096),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, (-1, 512 * 4 * 4))
        x = self.fcs(x)
        return x