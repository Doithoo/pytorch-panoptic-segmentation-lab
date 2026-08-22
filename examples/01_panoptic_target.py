"""Inspect Gaussian center and thing-only offset targets."""

import torch

from panoptic_segmenter.data.transforms import build_targets

semantic = torch.zeros((8, 8), dtype=torch.int64)
semantic[2:6, 2:6] = 1
instance = torch.zeros((8, 8), dtype=torch.int64)
instance[2:6, 2:6] = 1
target = build_targets(semantic, instance, (1,), center_sigma=1.0)
peak = torch.nonzero(target["center"] == target["center"].max()).tolist()
print("center peak:", peak)
print("heatmap mass:", round(float(target["center"].sum()), 4))
print("supervised offset pixels:", int(target["offset_mask"].sum()))
print("top-left thing offset:", target["offset"][:, 2, 2].tolist())
