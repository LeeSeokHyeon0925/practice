import torch

# Model_1
class SENet_v1(torch.nn.Module):
    def __init__(self, c_in, r):
        super(SENet_v1, self).__init__()
        c_mid = c_in // r

        self.avgp = torch.nn.AdaptiveAvgPool2d(1)
        self.maxp = torch.nn.AdaptiveMaxPool2d(1)

        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_mid),
            torch.nn.ReLU(),
            torch.nn.Linear(c_mid, c_in),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.avgp(x).squeeze()
        x_max = self.maxp(x).squeeze()

        x_se = x_avg * x_max

        x_se = self.fcs(x_se)

        x_se = torch.reshape(x_se, (x.shape[0], x.shape[1], 1, 1))
        x = x * x_se

        return x

# Model_2
class SENet_v2(torch.nn.Module):
    def __init__(self, c_in, r):
        super(SENet_v2, self).__init__()
        c_mid = c_in // r

        self.avgp = torch.nn.AdaptiveAvgPool2d(1)
        self.maxp = torch.nn.AdaptiveMaxPool2d(1)

        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_mid),
            torch.nn.ReLU(),
            torch.nn.Linear(c_mid, c_in),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.avgp(x).squeeze()
        x_max = self.maxp(x).squeeze()

        x_se = x_avg + x_max

        x_se = self.fcs(x_se)

        x_se = torch.reshape(x_se, (x.shape[0], x.shape[1], 1, 1))
        x = x * x_se

        return x

# Model_3
class SENet_v3(torch.nn.Module):
    def __init__(self, c_in, r):
        super(SENet_v3, self).__init__()
        c_mid = c_in // r

        self.avgp = torch.nn.AdaptiveAvgPool2d(1)
        self.maxp = torch.nn.AdaptiveMaxPool2d(1)

        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_mid),
            torch.nn.ReLU(),
            torch.nn.Linear(c_mid, c_in),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.avgp(x).squeeze()
        x_max = self.maxp(x).squeeze()

        x_avg = self.fcs(x_avg)
        x_max = self.fcs(x_max)

        x_se = x_avg * x_max

        x_se = torch.reshape(x_se, (x.shape[0], x.shape[1], 1, 1))
        x = x * x_se

        return x

# Model_4
class SENet_v4(torch.nn.Module):
    def __init__(self, c_in, r):
        super(SENet_v4, self).__init__()
        c_mid = c_in // r

        self.avgp = torch.nn.AdaptiveAvgPool2d(1)
        self.maxp = torch.nn.AdaptiveMaxPool2d(1)

        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_mid),
            torch.nn.ReLU(),
            torch.nn.Linear(c_mid, c_in),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.avgp(x).squeeze()
        x_max = self.maxp(x).squeeze()

        x_avg = self.fcs(x_avg)
        x_max = self.fcs(x_max)

        x_se = x_avg + x_max

        x_se = torch.reshape(x_se, (x.shape[0], x.shape[1], 1, 1))
        x = x * x_se

        return x

# SENet
class SENet(torch.nn.Module):
    def __init__(self, c_in, r):
        super(SENet, self).__init__()
        c_mid = c_in // r

        # global pooling
        self.avgp = torch.nn.AdaptiveAvgPool2d(1)

        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_mid),
            torch.nn.ReLU(),
            torch.nn.Linear(c_mid, c_in),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.avgp(x).squeeze()

        x_se = self.fcs(x_avg)

        x_se = torch.reshape(x_se, (x.shape[0], x.shape[1], 1, 1))
        x = x * x_se

        return x