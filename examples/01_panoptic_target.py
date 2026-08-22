"""Inspect center and offset targets for one synthetic sample."""

import torch

from panoptic_segmenter.data.transforms import build_targets

semantic = torch.zeros((8, 8), dtype=torch.int64)
semantic[2:6, 2:6] = 1
instance = torch.zeros((8, 8), dtype=torch.int64)
instance[2:6, 2:6] = 1
target = build_targets(semantic, instance, (1,))
print("center pixels:", torch.nonzero(target["center"]).tolist())
print("center count:", int(target["center"].sum()))
print("offset shape:", tuple(target["offset"].shape))
