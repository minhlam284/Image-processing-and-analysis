import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import os
# class DenseLayer(nn.Module):
#     def __init__(self, in_channels, growth_rate):
#         super(DenseLayer, self).__init__()
#         self.bn = nn.BatchNorm2d(in_channels)
#         self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
    
#     def forward(self, x):
#         out = self.conv(F.relu(self.bn(x)))
#         return torch.cat([x, out], 1)

# class DenseBlock(nn.Module):
#     def __init__(self, num_layers, in_channels, growth_rate):
#         super(DenseBlock, self).__init__()
#         layers = []
#         for i in range(num_layers):
#             layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
#         self.block = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.block(x)

# class TransitionLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(TransitionLayer, self).__init__()
#         self.bn = nn.BatchNorm2d(in_channels)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.pool = nn.AvgPool2d(2, stride=2)
    
#     def forward(self, x):
#         x = self.conv(F.relu(self.bn(x)))
#         return self.pool(x)

# class DenseNet(nn.Module):
#     def __init__(self, num_blocks, growth_rate=32, num_classes=10):
#         super(DenseNet, self).__init__()
#         num_layers_per_block = [6, 12, 24, 16]  # Default for DenseNet-121
#         in_channels = 64
        
#         self.init_conv = nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
#         self.init_bn = nn.BatchNorm2d(in_channels)
#         self.init_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         blocks = []
#         for i in range(num_blocks):
#             blocks.append(DenseBlock(num_layers_per_block[i], in_channels, growth_rate))
#             in_channels += num_layers_per_block[i] * growth_rate
#             if i != num_blocks - 1:
#                 blocks.append(TransitionLayer(in_channels, in_channels // 2))
#                 in_channels //= 2
        
#         self.features = nn.Sequential(*blocks)
#         self.final_bn = nn.BatchNorm2d(in_channels)
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(in_channels, num_classes)
    
#     def forward(self, x):
#         x = self.init_pool(F.relu(self.init_bn(self.init_conv(x))))
#         x = self.features(x)
#         x = self.global_pool(F.relu(self.final_bn(x)))
#         x = torch.flatten(x, 1)
#         return self.fc(x)

# # Khởi tạo mô hình
# model = DenseNet(num_blocks=4, growth_rate=32, num_classes=10)
# print(model)


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dir = "/Users/kaiser_1/Documents/GitHub/Image-processing-and-analysis/Image Classification/single_chromosome"  # Thay đổi đường dẫn nếu cần
train_dataset = ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
test_dataset = ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

print(train_dataset)


import torchvision.models as models

model = models.densenet121(pretrained=True)