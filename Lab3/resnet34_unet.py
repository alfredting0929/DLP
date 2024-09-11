import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def make_layer(block, in_channels, out_channels, blocks, stride=1):
    downsample = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * block.expansion),
        )

    layers = []
    layers.append(block(in_channels, out_channels, stride, downsample))
    in_channels = out_channels * block.expansion
    for _ in range(1, blocks):
        layers.append(block(in_channels, out_channels))

    return nn.Sequential(*layers)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handling spatial size mismatch
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x

class ResNet34UNet(nn.Module):
    def __init__(self):
        super(ResNet34UNet, self).__init__()
        self.initial = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True))
        
        # Encoder (ResNet34)                                        
        self.layer1 = make_layer(BasicBlock, 64, 64, 3)
        self.layer2 = make_layer(BasicBlock, 64, 128, 4, stride=2)
        self.layer3 = make_layer(BasicBlock, 128, 256, 6, stride=2)
        self.layer4 = make_layer(BasicBlock, 256, 512, 3, stride=2)

        # Decoder
        self.up1 = DecoderBlock(512, 256, 256)
        self.up2 = DecoderBlock(256, 128, 128)
        self.up3 = DecoderBlock(128, 64, 64)
        self.up4 = DecoderBlock(64, 64, 64)

        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.initial(x)
        # print(x1.size())
        x2 = self.layer1(x1)
        # print(x2.size())
        x3 = self.layer2(x2)
        # print(x3.size())
        x4 = self.layer3(x3)
        # print(x4.size())
        x5 = self.layer4(x4)
        # print(x5.size())

        x = self.up1(x5, x4)
        # print(x.size())
        x = self.up2(x, x3)
        # print(x.size())
        x = self.up3(x, x2)
        # print(x.size())
        x = self.up4(x, x1)
        # print(x.size())

        out = self.outc(x)
        out = self.bn(out)
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    model = ResNet34UNet()
    x = torch.randn(1,3,256,256)
    output = model(x)
    print(f'before {x.size()}')
    print(f'after {output.size()}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")