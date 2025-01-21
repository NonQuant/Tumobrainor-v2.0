import torch.nn as nn
from torchvision import models
from attention_block import SqueezeExciteBlock


class AttentionResNet50(nn.Module):
    def __init__(self, num_classes=4, freeze_backbone=True):
        super(AttentionResNet50, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        # freeze backbone to train only attention + final layers
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Remove the original avgpool + fc
        self.features = nn.Sequential(
            *list(self.resnet.children())[:-2]  # up to layer4
        )

        # Insert the SE attention block (after the last conv stage)
        in_channels = 2048
        self.attn = SqueezeExciteBlock(in_channels, reduction=16)

        # New global pooling & classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x shape: (N, 3, H, W)
        x = self.features(x)  # -> (N, 2048, H', W')
        x = self.attn(x)  # SE attention
        x = self.pool(x)  # -> (N, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten -> (N, 2048)
        x = self.fc(x)  # -> (N, num_classes)
        return x
