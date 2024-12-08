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

class UNet(torch.nn.Module):
    def __init__(self, num_class):
        super(UNet, self).__init__()

        self.enc_b1 = torch.nn.Sequential(
            conv_bn_relu(3, 64),
            conv_bn_relu(64, 64),
        )

        self.enc_b2 = torch.nn.Sequential(
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
        )

        self.enc_b3 = torch.nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
        )

        self.enc_b4 = torch.nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
        )

        self.enc_b5 = torch.nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
        )

        self.dec_b4 = torch.nn.Sequential(
            conv_bn_relu(1024, 512),
            conv_bn_relu(512, 512),
        )

        self.dec_b3 = torch.nn.Sequential(
            conv_bn_relu(768, 256),
            conv_bn_relu(256, 256),
        )

        self.dec_b2 = torch.nn.Sequential(
            conv_bn_relu(384, 128),
            conv_bn_relu(128, 128),
        )

        self.dec_b1 = torch.nn.Sequential(
            conv_bn_relu(192, 64),
            conv_bn_relu(64, 64),
        )

        self.maxp = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.up2x = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.predct = torch.nn.Conv2d(64, num_class, kernel_size=(3, 3))

    def forward(self, x):
        x_b1 = self.enc_b1(x) # [b, 512, 256, 256]

        x_b2 = self.maxp(x_b1)
        x_b2 = self.enc_b2(x_b2) # [b, 512, 128, 128]

        x_b3 = self.maxp(x_b1)
        x_b3 = self.enc_b3(x_b3) # [b, 512, 64, 64]

        x_b4 = self.maxp(x_b3)
        x_b4 = self.enc_b4(x_b4) # [b, 512, 32, 32]

        x_b5 = self.maxp(x_b4)
        x_b5 = self.enc_b5(x_b5) # [b, 512, 16, 16]

        x = self.up2x(x_b5) # [b, 512, 32, 32]
        x = torch.cat([x, x_b4], dim=1) # [b, 1024, 32, 32]

        x = self.dec_b4(x) #  [b, 512, 32, 32]

        x = self.up2x(x)  # [b, 512, 64, 64]
        x = torch.cat([x, x_b3], dim=1)  # [b, 768, 64, 64]

        x = self.dec_b3(x)  # [b, 256, 64, 64]

        x = self.up2x(x)  # [b, 256, 128, 128]
        x = torch.cat([x, x_b2], dim=1)  # [b, 384, 128, 128]

        x = self.dec_b2(x)  # [b, 128, 128, 128]

        x = self.up2x(x)  # [b, 128, 256, 256]
        x = torch.cat([x, x_b1], dim=1)  # [b, 192, 256, 256]

        x = self.dec_b1(x)  # [b, 64, 256, 256]
        x = self.predct(x)

        return x