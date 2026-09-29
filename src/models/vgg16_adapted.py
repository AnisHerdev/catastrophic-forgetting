"""
VGG16Adapted Architecture with Dynamically Expandable Classification Head.
Utilizes pre-trained ImageNet VGG16 convolutional backbone with custom classifier head
and dynamic output dimension expansion for class-incremental continual learning.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16Adapted(nn.Module):
    """
    Adapted VGG16 model with pre-trained convolutional features,
    Global Average Pooling, and a customizable, expandable classifier head.
    """

    def __init__(
        self,
        num_classes: int = 25,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load VGG16 backbone
        if pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1
            base_vgg = vgg16(weights=weights)
        else:
            base_vgg = vgg16(weights=None)

        # Convolutional feature extractor (512 output channels at last conv stage)
        self.features = base_vgg.features
        
        # Adaptive pooling to handle 32x32 CIFAR-100 inputs cleanly to (Batch, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Compact classification head (512 -> 256 -> num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        # Initialize the custom classifier linear layers if not pre-trained
        self._init_classifier_weights()

    def _init_classifier_weights(self) -> None:
        """Initializes weights of the custom head using Kaiming normal."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def expand_classes(self, new_num_classes: int) -> None:
        """
        Dynamically expands the final classification layer to accommodate new classes,
        preserving previously learned weights and biases for existing classes.

        Args:
            new_num_classes: New total number of output classes.
        """
        if new_num_classes <= self.num_classes:
            return

        old_num_classes = self.num_classes
        old_fc: nn.Linear = self.classifier[3]
        device = next(self.parameters()).device

        # Create new linear layer with expanded output dimension
        new_fc = nn.Linear(old_fc.in_features, new_num_classes).to(device)

        # Initialize new layer weights
        nn.init.kaiming_normal_(new_fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(new_fc.bias, 0)

        # Copy existing learned weights and biases for previously seen classes
        with torch.no_grad():
            new_fc.weight[:old_num_classes, :] = old_fc.weight.data
            new_fc.bias[:old_num_classes] = old_fc.bias.data

        # Swap classifier final layer
        self.classifier[3] = new_fc
        self.num_classes = new_num_classes

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freezes or unfreezes the convolutional feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = not freeze

    def count_parameters(self) -> dict:
        """Returns the total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
