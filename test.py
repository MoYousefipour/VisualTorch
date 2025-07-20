import torch
import torch.nn as nn
from collections import defaultdict
from PIL import ImageFont
import visualtorch  # Replace visualkeras with visualtorch


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ZeroPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ZeroPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Instantiate model
model = VGG16()

color_map = defaultdict(dict)
color_map[nn.Conv2d]['fill'] = 'orange'
color_map[nn.ZeroPad2d]['fill'] = 'gray'
color_map[nn.Dropout]['fill'] = 'pink'
color_map[nn.MaxPool2d]['fill'] = 'red'
color_map[nn.Linear]['fill'] = 'green'
color_map[nn.Flatten]['fill'] = 'teal'

font = ImageFont.truetype("arial.ttf", 32)

# Visualize with visualtorch
visualtorch.layered_view(model, to_file='vgg16_torch.png')
