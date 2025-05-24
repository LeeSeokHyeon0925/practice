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

class FCN_32S(torch.nn.Module):
    def __init__(self, num_class):
        super(FCN_32S, self).__init__()
        self.vgg16 = torch.nn.Sequential(
            conv_bn_relu(3, 64),  # [b, 64, 256, 256]
            conv_bn_relu(64, 64),  # [b, 64, 256, 256]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 64, 128, 128]

            conv_bn_relu(64, 128),  # [b, 128, 128, 128]
            conv_bn_relu(128, 128),  # [b, 128, 128, 128]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 128, 64, 64]

            conv_bn_relu(128, 256),  # [b, 256, 64, 64]
            conv_bn_relu(256, 256),  # [b, 256, 64, 64]
            conv_bn_relu(256, 256),  # [b, 256, 64, 64]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 256, 32, 32]

            conv_bn_relu(256, 512),  # [b, 512, 32, 32]
            conv_bn_relu(512, 512),  # [b, 512, 32, 32]
            conv_bn_relu(512, 512),  # [b, 512, 32, 32]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 16, 16]

            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 8, 8]

            conv_bn_relu(512, 4096, 1, 0),  # [b, 4096, 8, 8]
            conv_bn_relu(4096, 4096, 1, 0),  # [b, 4096, 8, 8]
            torch.nn.Conv2d(4096, num_class, kernel_size=(1, 1)), # [b, num_class, 8, 8]
        )

        self.up = torch.nn.ConvTranspose2d(num_class, num_class, kernel_size=64, stride=32, padding=16) # x32

    def forward(self, x):
        x = self.vgg16(x) # [b, num_class, 8, 8]
        x = self.up(x) # [b, num_class, 256, 256]

        return x

class FCN_16S(torch.nn.Module):
    def __init__(self, num_class):
        super(FCN_16S, self).__init__()
        self.vgg16_part1 = torch.nn.Sequential(
            conv_bn_relu(3, 64),  # [b, 64, 256, 256]
            conv_bn_relu(64, 64),  # [b, 64, 256, 256]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 64, 128, 128]

            conv_bn_relu(64, 128),  # [b, 128, 128, 128]
            conv_bn_relu(128, 128),  # [b, 128, 128, 128]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 128, 64, 64]

            conv_bn_relu(128, 256),  # [b, 256, 64, 64]
            conv_bn_relu(256, 256),  # [b, 256, 64, 64]
            conv_bn_relu(256, 256),  # [b, 256, 64, 64]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 256, 32, 32]

            conv_bn_relu(256, 512),  # [b, 512, 32, 32]
            conv_bn_relu(512, 512),  # [b, 512, 32, 32]
            conv_bn_relu(512, 512),  # [b, 512, 32, 32]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 16, 16]
        )

        self.vgg16_part2 = torch.nn.Sequential(
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 8, 8]

            conv_bn_relu(512, 4096, 1, 0),  # [b, 4096, 8, 8]
            conv_bn_relu(4096, 4096, 1, 0),  # [b, 4096, 8, 8]
            torch.nn.Conv2d(4096, num_class, kernel_size=(1, 1)),
        )

        self.predic_part1 = torch.nn.Conv2d(512, num_class, kernel_size=(1, 1))

        self.up_p2 = torch.nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1) # x2
        self.up_p1 = torch.nn.ConvTranspose2d(num_class, num_class, kernel_size=32, stride=16, padding=8) # x16

    def forward(self, x):
        x = self.vgg16_part1(x)
        x_p1 = self.predic_part1(x) # [b, num_class, 16, 16]

        x = self.vgg16_part2(x)
        x = self.up_p2(x) # [b, num_class, 16, 16]
        x = x + x_p1

        x = self.up_p1(x) # [b, num_class, 256, 256]

        return x

class FCN_8S(torch.nn.Module):
    def __init__(self, num_class):
        super(FCN_8S, self).__init__()
        self.vgg16_part1 = torch.nn.Sequential(
            conv_bn_relu(3, 64),  # [b, 64, 256, 256]
            conv_bn_relu(64, 64),  # [b, 64, 256, 256]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 64, 128, 128]

            conv_bn_relu(64, 128),  # [b, 128, 128, 128]
            conv_bn_relu(128, 128),  # [b, 128, 128, 128]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 128, 64, 64]

            conv_bn_relu(128, 256),  # [b, 256, 64, 64]
            conv_bn_relu(256, 256),  # [b, 256, 64, 64]
            conv_bn_relu(256, 256),  # [b, 256, 64, 64]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 256, 32, 32]
        )

        self.vgg16_part2 = torch.nn.Sequential(
            conv_bn_relu(256, 512),  # [b, 512, 32, 32]
            conv_bn_relu(512, 512),  # [b, 512, 32, 32]
            conv_bn_relu(512, 512),  # [b, 512, 32, 32]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 16, 16]
        )

        self.vgg16_part3 = torch.nn.Sequential(
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            conv_bn_relu(512, 512),  # [b, 512, 16, 16]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # [b, 512, 8, 8]

            conv_bn_relu(512, 4096, 1, 0),  # [b, 4096, 8, 8]
            conv_bn_relu(4096, 4096, 1, 0),  # [b, 4096, 8, 8]
            torch.nn.Conv2d(4096, num_class, kernel_size=(1, 1)),
        )

        self.predic_part1 = torch.nn.Conv2d(256, num_class, kernel_size=(1, 1))
        self.predic_part2 = torch.nn.Conv2d(512, num_class, kernel_size=(1, 1))

        self.up_p3 = torch.nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1) # x2
        self.up_p2 = torch.nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1) # x2
        self.up_p1 = torch.nn.ConvTranspose2d(num_class, num_class, kernel_size=16, stride=8, padding=4) # x8

    def forward(self, x):
        x = self.vgg16_part1(x)
        x_p1 = self.predic_part1(x) # [b, num_class, 32, 32]

        x = self.vgg16_part2(x)
        x_p2 = self.predic_part2(x) # [b, num_class, 16, 16]

        x = self.vgg16_part3(x)
        x = self.up_p3(x) # [b, num_class, 16, 16]
        x = x + x_p2

        x = self.up_p2(x) # [b, num_class, 32, 32]
        x = x + x_p1

        x = self.up_p1(x) # [b, num_class, 256, 256]

        return x