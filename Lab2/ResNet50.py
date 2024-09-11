import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_feature, config, connect, downsampling=False):
        super().__init__()
        self.in_feature = in_feature
        self.config = config
        self.connect = connect
        if downsampling:
            # To achieve downsampling, using stride 2
            self.block = nn.Sequential(nn.Conv2d(in_channels=in_feature, out_channels=config[0], kernel_size=1),
                                       nn.BatchNorm2d(config[0]),
                                       nn.ReLU(True),
                                       nn.Conv2d(in_channels=config[0], out_channels=config[1], kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(config[1]),
                                       nn.ReLU(True),
                                       nn.Conv2d(in_channels=config[1], out_channels=config[2], kernel_size=1),
                                       nn.BatchNorm2d(config[2]))
        else:
            self.block = nn.Sequential(nn.Conv2d(in_channels=in_feature, out_channels=config[0], kernel_size=1),
                                       nn.BatchNorm2d(config[0]),
                                       nn.ReLU(True),
                                       nn.Conv2d(in_channels=config[0], out_channels=config[1], kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(config[1]),
                                       nn.ReLU(True),
                                       nn.Conv2d(in_channels=config[1], out_channels=config[2], kernel_size=1),
                                       nn.BatchNorm2d(config[2]))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Residual connect
        residual = self.connect(x)
        x = self.block(x)
        x += residual
        x = self.relu(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # First part of the ResNet50 (conv1): 7*7, 64 stride 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        # Second part of the ResNet50: 3*3 max pool, stride 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Conv2_x
        self.conv2 = self.generate_layer(num_layer=3, config=[64,64,256], in_feature=64, downsampling=False)
        # Conv3_x
        self.conv3 = self.generate_layer(num_layer=4, config=[128,128,512], in_feature=256, downsampling=True)
        # Conv4_x
        self.conv4 = self.generate_layer(num_layer=6, config=[256,256,1024], in_feature=512, downsampling=True)
        # Conv5_x
        self.conv5 = self.generate_layer(num_layer=3, config=[512,512,2048], in_feature=1024, downsampling=True)
        # Last part of the ResNet50: average pool, 1000-d fc, softmax
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        
    def generate_layer(self, num_layer, config, in_feature, downsampling):
        layer = []
        if downsampling:
            connect = nn.Sequential(nn.Conv2d(in_channels=in_feature, out_channels=config[-1], kernel_size=1, stride=2),
                                    nn.BatchNorm2d(config[-1]))
            layer.append(Bottleneck(in_feature=in_feature, config=config, connect=connect, downsampling=True))
        else:
            connect = nn.Sequential(nn.Conv2d(in_channels=in_feature, out_channels=config[-1], kernel_size=1, stride=1),
                                    nn.BatchNorm2d(config[-1]))
            layer.append(Bottleneck(in_feature=in_feature, config=config, connect=connect))

        for _ in range(num_layer - 1):
            connect = nn.Sequential(nn.Conv2d(in_channels=config[-1], out_channels=config[-1], kernel_size=1, stride=1),
                                    nn.BatchNorm2d(config[-1]))
            layer.append(Bottleneck(in_feature=config[-1], config=config, connect=connect))

        return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':
    model = ResNet50(num_classes=100)
    example_input = torch.randn((1,3,224,224))
    print(model(example_input))