# YOLOv8 网络结构与数学原理详解

> 本文档配合 [README.md](../README.md) 的第 5 节使用, 深入讲 YOLOv8 的网络结构、
> Loss 数学公式和关键算子. 适合想理解源码、调超参或做改进的研究者.

---

## 目录

- [1. 整体结构](#1-整体结构)
- [2. Backbone: CSPDarknet + C2f](#2-backbone-cspdarknet--c2f)
- [3. Neck: PANet / FPN](#3-neck-panet--fpn)
- [4. Head: Anchor-Free Decoupled](#4-head-anchor-free-decoupled)
- [5. Loss 函数](#5-loss-函数)
- [6. TaskAlignedAssigner 标签分配](#6-taskalignedassigner-标签分配)
- [7. 训练时的数据增强](#7-训练时的数据增强)
- [8. 推理时的后处理](#8-推理时的后处理)

---

## 1. 整体结构

YOLOv8 是 **single-stage detector**, 输入一张 640×640×3 的图, 输出 8400 个预测.

```
┌────────────────────────────────────────────────────────────────────┐
│  Input: 640×640×3 (RGB)                                            │
└────────────────────────┬───────────────────────────────────────────┘
                         │
                ┌────────▼────────┐
                │   Backbone      │   CSPDarknet (modified)
                │   (extract      │
                │    features)    │
                └────┬─────┬──────┘
            P3 (80×80)│     │P4 (40×40)   P5 (20×20)
                     │     │     │
                     ▼     ▼     ▼
                ┌────────────────┐
                │   Neck         │   PANet (Path Aggregation)
                │   (multi-scale │
                │    fusion)     │
                └───┬──────┬─────┘
            P3'      │      │  P5'
                     ▼      ▼
                ┌────────────────┐
                │   Head         │   Decoupled, Anchor-free
                │   (predict)    │
                └────────────────┘
                     │  │  │
                     ▼  ▼  ▼
              80×80 40×40 20×20  = 8400 个 grid 输出
```

每个输出 grid 包含:
- 4 个 bbox 回归值(l, t, r, b 距离)
- 1 个类别概率(单类时; 多类时是 `num_classes` 个)
- DFL 分布(用于学习 bbox 边界的不确定性)

---

## 2. Backbone: CSPDarknet + C2f

### 2.1 整体形状

```python
# 简化版伪代码
x = Conv(img, 3→32, k=3, s=1)        # 640
x = Conv(x, 32→64, k=3, s=2)        # 320  (P1/2)
x = Conv(x, 64→128, k=3, s=2)       # 160  (P2/4)
x = C2f(x, 128)
x = Conv(x, 128→256, k=3, s=2)      # 80   (P3/8)  → 给 Neck
x = C2f(x, 256)
x = Conv(x, 256→512, k=3, s=2)      # 40   (P4/16) → 给 Neck
x = C2f(x, 512)
x = Conv(x, 512→1024, k=3, s=2)     # 20   (P5/32) → 给 Neck
x = C2f(x, 1024)
x = SPPF(x, 1024)
```

### 2.2 C2f 模块(改进自 YOLOv5 的 C3)

```
       input
         │
    ┌────┴────┐
    │         │
   Conv    Identity
    │         │
   ...      (split)
    │         │
    └────┬────┘
       concat
         │
       Conv
```

C2f 用 **2 个分支** 取代 C3 的 1 个分支, 让梯度流更密集. 加上 `n` 个
Bottleneck 串联, 表达力更强.

**Bottleneck 内部**: `1×1 Conv → 3×3 Conv`, 带可选残差.

### 2.3 SPPF (Spatial Pyramid Pooling Fast)

```python
# 一句话: 3 个 5x5 MaxPool 串联, 等价于 5/9/13 三个 kernel 的并行 SPP, 但更快
y1 = MaxPool(x, k=5)
y2 = MaxPool(y1, k=5)
y3 = MaxPool(y2, k=5)
out = Conv(concat([x, y1, y2, y3]), ...)
```

作用: **扩大感受野**, 让网络"看到"整个图的全局信息.

---

## 3. Neck: PANet / FPN

YOLOv8 的 Neck 是 **PANet (Path Aggregation Network)** 的简化版:

```
   P3 ────────────► Conv ───┐
                              Upsample ┐
   P4 ──── Conv ─── Concat ──┘         │
                                       Conc
   P5 ──── Conv ─── Upsample ─────────►
```

核心思想:
- **FPN 方向(自顶向下)**: 把高层(语义强但分辨率低)的特征上采样后, 跟低层拼接,
  补全小目标的语义信息.
- **PAN 方向(自底向上)**: 再把低层(细节强)的特征下采样后, 跟高层拼接,
  补全大目标的位置信息.

这样每个尺度的 Head 都同时拥有"细节"和"语义".

---

## 4. Head: Anchor-Free Decoupled

### 4.1 Decoupled(解耦)

老的 YOLOv5 是 **Coupled Head**: 分类和回归共享卷积. YOLOv8 拆成两条独立的卷积分支:

```
       P3' ──►┬─► bbox branch (4 ch)
              ├─► cls branch (num_classes ch)
              └─► dfl branch (4 × reg_max ch, 默认 reg_max=16)
```

好处:
- 分类和回归是 **不同性质的任务**(分类是离散, 回归是连续), 解耦后各自学最优
- 收敛更快, 精度更高

### 4.2 Anchor-Free

不再预设 anchor 框, 每个 grid 直接预测:

```
       ┌─────────── grid (x_c, y_c) ───────────┐
       │                                       │
       │  l = grid 到左边界距离                 │
       │  t = grid 到上边界距离                 │
       │  r = grid 到右边界距离                 │
       │  b = grid 到下边界距离                 │
       │                                       │
       │  bbox = [x_c - l, y_c - t,            │
       │         x_c + r, y_c + b]             │
       └───────────────────────────────────────┘
```

训练时, 只要 grid 落在某个 GT 框内, 就让它学这个框的 (l, t, r, b).

### 4.3 DFL: Distribution Focal Loss

bbox 回归值不是一个标量, 而是一个 **离散概率分布**, 默认 `reg_max=16`, 即每个
l/t/r/b 都用 16 个 bin 表示.

例如预测 `l` 时, 输出 16 个 logit, softmax 后得到分布 `P(l=0), P(l=1), ..., P(l=15)`,
最终回归值 `l_pred = sum_i (i × P(l=i))`.

**为什么这么搞?** 真实数据的边界框有固有的不确定性(模糊边缘、遮挡),
用分布建模比单点回归更鲁棒. DFL Loss 强制分布集中在 GT 附近:

```
DFL = -((y_{i+1} - y) * log P(y_i) + (y - y_i) * log P(y_{i+1}))
其中 y 是 GT, y_i ≤ y ≤ y_{i+1}
```

---

## 5. Loss 函数

总 Loss:

```
L_total = λ_box · L_box + λ_cls · L_cls + λ_dfl · L_dfl

默认 λ_box = 7.5, λ_cls = 0.5, λ_dfl = 1.5
```

### 5.1 Bbox Loss: CIoU

CIoU = Complete IoU, 在普通 IoU 基础上加了 **中心距离** 和 **长宽比**:

```
CIoU = IoU - (ρ²(b, b_gt) / c²) - αv

其中:
  ρ²(b, b_gt) = 预测框中心与 GT 中心的欧氏距离²
  c           = 两个框最小外接矩形的对角线长度
  v           = (4/π²) · (arctan(w_gt/h_gt) - arctan(w/h))²
  α           = v / ((1 - IoU) + v)
```

`L_box = 1 - CIoU`.

### 5.2 Cls Loss: Varifocal Loss (VFL)

VFL 是 Focal Loss 的变种, 对正负样本不对称处理:

```
VFL = {
  -q · (q · log p + (1-q) · log(1-p))   if 正样本 (q = IoU·label)
  -α · p^γ · log(1-p)                    if 负样本
}

默认 α=0.75, γ=2.0
```

**关键**: 正样本的 target `q` 不是 0/1, 而是 `IoU × label`, 这样分类得分
高的样本天然有更大的 IoU, 实现分类和定位的"协同".

### 5.3 DFL Loss

如 4.3 节, 让 bbox 边界分布靠近 GT 离散值.

---

## 6. TaskAlignedAssigner 标签分配

### 6.1 问题

每个 grid 都会预测一个框, 但只有少数 grid 是"对"的(对应某个 GT).
怎么挑出这些正样本?

老方法(如 YOLOv5): 按 IoU 阈值, 大于 0.5 就是正样本.
缺点: 不考虑分类得分, 可能选到"分得对但定位差"的样本.

### 6.2 TaskAligned

YOLOv8 用 **TaskAlignedAssigner**, 定义"对齐度":

```
t = s^α · u^β

s = 分类得分
u = 预测框与 GT 的 IoU
α, β 默认都是 0.5
```

意思: 一个 grid 同时 **分类对** 和 **位置对** 才是好样本.

### 6.3 Top-K 选择

对每个 GT, 算所有 grid 的 `t`, 取 top-K(默认 K=13) 作为正样本,
其余 grid 强制为负样本.

---

## 7. 训练时的数据增强

`ultralytics/cfg/default.yaml` 里默认:

| 增强 | 概率/参数 | 作用 |
|---|---|---|
| **Mosaic** | 1.0 | 把 4 张图拼成 1 张, 模拟不同上下文 |
| **MixUp** | 0.15(仅 s/m/l/x) | 两张图加权混合 |
| **HSV(h/s/v)** | 0.015/0.7/0.4 | 色调/饱和度/明度扰动 |
| **Flip lr** | 0.5 | 水平翻转 |
| **Flip ud** | 0.0 | 垂直翻转(默认关) |
| **Scale** | 0.5 | 缩放 |
| **Translate** | 0.1 | 平移 |
| **Copy-Paste** | 0.0(only seg) | 复制粘贴 |

**注意**: 训练最后 10 个 epoch, **Mosaic 和 MixUp 自动关闭**, 让模型
在原始分布上 fine-tune, 提升最终精度.

---

## 8. 推理时的后处理

### 8.1 Letterbox

输入图先 **letterbox** 等比缩放到 640×640(短边补灰边), 而不是简单 resize,
保持长宽比, 不变形.

### 8.2 Forward

`model.predict()` 直接 forward 一次, 输出 8400 个预测:

```
P3 (80×80)  = 6400 个
P4 (40×40)  = 1600 个
P5 (20×20)  =  400 个
-----------------------
合计         = 8400 个
```

### 8.3 过滤

```
preds = forward(img)
mask = preds.conf > 0.25         # 置信度阈值
preds = preds[mask]
```

### 8.4 NMS (Non-Maximum Suppression)

```
对每个类别:
  按置信度从高到低排序
  保留最高的, 把与它 IoU > 0.7 的全部丢掉
  重复
```

### 8.5 坐标还原

letterbox 后的坐标 → 减去 pad → 乘回原图缩放比 → 得到原图上的坐标.

---

## 9. 推荐阅读

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLOv8 原论文的解读(知乎)](https://zhuanlan.zhihu.com/p/598545844)
- [DFL 原论文: Generalized Focal Loss](https://arxiv.org/abs/2006.04388)
- [CIoU 原论文](https://arxiv.org/abs/2005.03572)
- [VarifocalNet](https://arxiv.org/abs/2008.13367)

---

## 10. 源码导读

`ultralytics/cfg/models/v8/yolov8.yaml` 定义了网络结构(`nc` 改类别数即可).
关键文件:

```
ultralytics/
├── cfg/models/v8/yolov8.yaml          ← 网络结构定义
├── cfg/datasets/coco.yaml             ← 数据集类名
├── cfg/default.yaml                   ← 所有默认超参
├── nn/
│   ├── tasks.py                       ← 模型构建
│   ├── modules/
│   │   ├── block.py                   ← C2f, SPPF 等
│   │   ├── conv.py                    ← Conv, Bottleneck
│   │   └── head.py                    ← Detect head
├── utils/
│   ├── loss.py                        ← 各种 Loss
│   ├── tal.py                         ← TaskAlignedAssigner
│   └── ops.py                         ← NMS 等
└── engine/
    ├── trainer.py
    ├── predictor.py
    └── validator.py
```

把这几个文件读一遍, 基本就理解 YOLOv8 全部原理了.
