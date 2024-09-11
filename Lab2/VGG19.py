import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # First block, output: 112
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        # Second block, output: 56
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        # Third block, output 28
        self.block3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        # Forth block, output: 14
        self.block4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        # Fifth block, output: 7
        self.block5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Linear(in_features=512*7*7, out_features=4096, bias=False),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=4096, out_features=4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=4096, out_features=num_classes))

        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = VGG19(num_classes=100)
    example_input = torch.randn((1,3,224,224))
    print(model(example_input))