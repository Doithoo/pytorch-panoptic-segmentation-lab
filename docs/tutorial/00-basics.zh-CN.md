# 00：全景分割 Tensor 与 ID

[English](00-basics.md) | [学习路线](learning-path.zh-CN.md)

全景分割对每个像素回答两个问题：它属于什么类别；若为 thing，它属于哪个对象。

项目的 ground truth：

- `semantic[y,x]`：连续 class ID，void 为 `255`。
- `instance[y,x]`：thing 像素使用图像内正整数对象 ID，其余为 `0`。

instance ID 本身不含类别语义。`(semantic=1, instance=7)` 共同定义一个区域；另一张图中的 ID 7 与它无关。一个 thing 实例不能跨越多个语义类。

模型输出：

```text
semantic logits [B,C,H,W]
center logits   [B,1,H,W]
offset          [B,2,H,W]
```

交叉熵学习语义类别；Gaussian heatmap 把对象中心变成可学习的局部峰值；thing 像素 `(y,x)` 的 offset `[dy,dx]` 指向实例均值 `(cy,cx)`，即 `dy=cy-y`、`dx=cx-x`。

解码时先对 semantic 做 `argmax`，center NMS 找到数量受限的峰值，再让 thing 像素加上预测 offset，加入同类别最近中心。stuff 不需要实例 ID。

不要用 `argmax` 后的值训练交叉熵，不要对标签做 bilinear resize，不要跨图复用实例身份，也不要把单独 semantic mask 当作完整 panoptic 输出。
