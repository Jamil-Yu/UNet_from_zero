import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # following a batch-norm layer, bias is unnecessary
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # inplace=True: save memory. The inplace argument to nn.ReLU controls whether the
            # layer modifies the input in place, or returns a new tensor with the modified output. If inplace=True,
            # then the input is modified in place, without allocating any additional memory. This can be useful if
            # you are trying to save memory, or if you need to modify the input multiple times.
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    # 实现了一个双卷积层，这个双卷积层是UNet中的基本组件，也是UNet的核心组件。
    # 一个双卷积层包含两个卷积层，中间加入了一个ReLU激活函数，同时在卷积层之间加入了一个BatchNorm2d层。
    # 为什么要加入BatchNorm2d层呢？因为在卷积层之间加入BatchNorm2d层可以加快训练速度，同时也可以提高模型的泛化能力。


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # The nn.ModuleList class is a subclass of nn.Module that is used to create a list of modules. Modules are
        # the basic building blocks of neural networks, and the nn.ModuleList class allows you to create a list of
        # modules that can be used together to create a more complex neural network.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # to make input size equal to output size, the input size should be divisible by 2^4

        # down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        # up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)  # upsample
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            # ups: ConvTranspose2d + DoubleConv + ConvTranspose2d + DoubleConv +
            # ConvTranspose2d + DoubleConv + ConvTranspose2d + DoubleConv
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)  # save the output of each down layer, going to be used in up layers
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse the list
        for idx in range(0, len(self.ups), 2):  # 0, 2, 4, 6
            x = self.ups[idx](x)  # upsample
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:  # if the size of x is not equal to the size of skip_connection, resize x
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)  # link the upsampled x and the skip_connection by channel
            x = self.ups[idx + 1](concat_skip)
        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape
    print("Success!")


if __name__ == "__main__":
    test()