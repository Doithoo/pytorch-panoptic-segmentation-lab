"""Build a single model batch and print its three output heads."""

import torch

from panoptic_segmenter.models import create_model

model = create_model("panoptic_unet_small", in_channels=3, num_classes=3, base_channels=8)
outputs = model(torch.randn(2, 3, 128, 128))
for name, value in outputs.items():
    print(name, tuple(value.shape))
