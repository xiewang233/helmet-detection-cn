# YOLOv8 安全帽检测 · RTX 5060 Blackwell 实战

> 在 NVIDIA RTX 5060 Laptop(Blackwell sm_120)上从零搭建的 YOLOv8 安全帽检测项目.
> **helmet mAP50 = 0.962,head mAP50 = 0.956**(100 epoch 完整训练, RTX 5060 实测),达到生产可用水平.

📌 **只想用模型?看 [USAGE.md](USAGE.md)** · 想懂原理?看 [ARCHITECTURE.md](docs/ARCHITECTURE.md)

![status](https://img.shields.io/badge/status-ready-brightgreen)
![python](https://img.shields.io/badge/python-3.11-blue)
![pytorch](https://img.shields.io/badge/torch-2.11.0%2Bcu128-orange)
![gpu](https://img.shields.io/badge/GPU-RTX%205060%20sm__120-success)
![license](https://img.shields.io/github/license/xiewang233/helmet-detection-cn)
![stars](https://img.shields.io/github/stars/xiewang233/helmet-detection-cn?style=social)

---

## ✨ 核心创新点

本项目不是简单的 `pip install ultralytics` 跑通,而是解决了以下 6 个真实工程难题:

### 1. Blackwell sm_120 架构适配(业内少见的完整方案)

RTX 50 系列是 NVIDIA 2025 年的新架构,主流 PyTorch 教程都跑不动. 本项目首次系统化解决:

- 老版本 PyTorch 报 `CUDA error: no kernel image is available for execution on the device`
- 用 **PyTorch 2.11.0 + CUDA 12.8** 轮子(`+cu128`),原生支持 sm_120
- 实测 FP16 矩阵乘 10.7 TFLOPS,Blackwell 满血

### 2. 混合镜像源策略(国内速度 + 国际兼容性双赢)

```yaml
conda → 清华 TUNA 镜像
pip   → 清华 TUNA 镜像(通用包)
PyTorch cu128 → 官方源 download.pytorch.org(清华源不镜像)
```

只用清华源 → 装不到 cu128 轮子 → 跑不动 RTX 5060
只用官方源 → 国内下载几小时
混合策略 → 完美兼容

### 3. 完整工业级工具链(11 个脚本,不是 notebook 拼凑)

`自检 → 数据下载 → 格式转换 → 训练 → 图片/视频/摄像头推理 → 性能压测 → ONNX/TensorRT 导出` 一条龙.
每个脚本都带 argparse,可独立调用,可串成 pipeline.

### 4. Windows multiprocessing OOM 工程解决方案

YOLOv8 在 100 epoch 末尾关闭 Mosaic 增强时会重启 dataloader,Windows 上 8 worker 同时 spawn
会触发 MemoryError. 本项目通过 `workers=4` + `cache='disk'` 解决,经验写入 README.

### 5. 多路实时训练监控体系

- TQDM 进度条(终端)
- `results.csv` 每 epoch 落盘(`tail -f` 实时看)
- `results.png` 每 epoch 重画
- `nvidia-smi -l 1` GPU 状态
- 配合 cron 定时汇报

### 6. 生产级部署就绪

- mAP50 ≈ 0.95 的真实 helmet/head 检测器
- 支持导出 ONNX / TensorRT,FP16 加速 2-3 倍
- 提供 FastAPI 集成示例
- RTSP 网络摄像头直接接入

---

## 📸 效果展示(实测截图)

### GPU / 环境自检

`python src/check_gpu.py` — 验证 PyTorch cu128 能正确驱动 RTX 5060 Blackwell.

![GPU 自检](screenshots/00_check_gpu.png)

### HardHat 训练曲线(100 epoch 完整训练)

`runs/detect/hardhat_100ep/results.png`

![训练曲线](screenshots/14_hardhat_train_100ep.png)

### HardHat 评估指标(100 epoch 完整训练)

| 类别 | Precision | Recall | **mAP50** | mAP50-95 |
|---|---|---|---|---|
| **helmet**(戴帽) | 0.925 | 0.930 | **0.962** | 0.626 |
| **head**(没戴) | 0.876 | 0.936 | **0.956** | 0.631 |
| 综合(含 person) | 0.628 | 0.629 | 0.649 | 0.424 |

混淆矩阵:

![混淆矩阵](screenshots/15_hardhat_confusion.png)

PR 曲线:

![PR 曲线](screenshots/16_hardhat_pr_curve.png)

### 真实工地推理(测试集)

下面 5 张是用本次训练的 `best.pt`(helmet mAP50=0.962 / head 0.956)在测试集抽图推理的结果.
**绿框 = helmet(戴帽), 红框 = head(未戴)**:

![pic1](screenshots/18_inference_pic1.png)
*19 个 head —— 大批未戴安全帽(告警场景)*

![pic2](screenshots/19_inference_pic2.png)
*21 个 head —— 未戴场景*

![pic3](screenshots/20_inference_pic3.png)
*23 head + 2 helmet —— 混合, 未戴为主*

![pic4](screenshots/21_inference_pic4.png)
*60 个 helmet —— 全员规范佩戴*

![pic5](screenshots/22_inference_pic5.png)
*41 个 helmet —— 规范佩戴*

更多推理结果见 `screenshots/` 目录(18-22 号图).

### RTX 5060 性能压测

`python src/benchmark.py`

![benchmark](screenshots/09_benchmark.png)

| 模型 | 精度 | 延迟 | FPS | 显存 |
|---|---|---|---|---|
| yolov8n | FP32 | 7.59 ms | **131.7** | 37 MB |
| yolov8n | FP16 | 7.68 ms | 130.2 | **28 MB** |
| yolov8s | FP32 | 7.75 ms | 129.0 | 97 MB |
| yolov8s | FP16 | **7.35 ms** | **136.1** | **54 MB** |

> FP32 vs FP16: yolov8n 二者速度持平(小模型非瓶颈), FP16 显存省 24%;
> yolov8s 用 FP16 既快 5% 又省显存 44%, **部署推荐 yolov8s + FP16**.
> 柱状图见 `screenshots/09_benchmark.png`(FP32) / `screenshots/24_benchmark_fp16.png`(FP16).

---

## 🚀 快速开始

### 一键安装(双击)

1. 进项目根目录
2. 右键 `install.bat` → 以管理员身份运行
3. 等 10-30 分钟,自动完成 conda 安装 + 环境创建 + PyTorch + 依赖 + 自检

### 手动安装

```bash
# 1. 创建 conda 环境
conda create -n yolov8 python=3.11 -y
conda activate yolov8

# 2. 装 PyTorch(必须用官方源 cu128,RTX 5060 必须)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 装其他依赖(走清华源,快)
pip install -r requirements.txt

# 4. 自检
python src/check_gpu.py
```

### 5 秒上手

```bash
conda activate yolov8
python src/detect_image.py --model models/hardhat_best.pt --source 你的图.jpg --save
# 结果:data/output/image_pred/你的图.jpg
```

详细用法见 **[USAGE.md](USAGE.md)**.

---

## 📁 项目结构

```
YOLOv8-Helmet-Detection/
├── README.md                  ← 本文档(项目概览)
├── USAGE.md                   ← 使用说明书(详细操作手册)
├── LICENSE                    ← MIT
├── docs/ARCHITECTURE.md       ← YOLOv8 网络结构详解
├── screenshots/               ← 本次复现实测截图(训练曲线/混淆矩阵/PR/推理/压测)
├── configs/
│   ├── helmet.yaml            ← 数据集类名/路径
│   └── train_params.yaml      ← 训练超参
├── Makefile                   ← 常用命令入口 (check/prepare/train/benchmark/export)
├── src/
│   ├── check_gpu.py           ← GPU/CUDA 自检
│   ├── download_data.py       ← 下载 SHWD 数据集
│   ├── prepare_data.py        ← SHWD VOC→YOLO 转换
│   ├── prepare_hardhat.py     ← Kaggle HardHat VOC→YOLO 转换
│   ├── dataset_utils.py       ← 数据准备公共工具(VOC→YOLO/划分/链接, 去重)
│   ├── train.py               ← 训练入口
│   ├── detect_image.py        ← 图片推理
│   ├── detect_video.py        ← 视频推理
│   ├── detect_camera.py       ← 摄像头实时
│   ├── benchmark.py           ← 性能压测 (CSV+柱状图落盘)
│   ├── export_model.py        ← ONNX/TensorRT 导出
│   ├── render_terminal.py     ← 终端输出渲染成 PNG
│   └── utils.py
├── models/hardhat_best.pt     ← 训练好的权重(gitignore)
├── datasets/                  ← 数据集(gitignore)
└── runs/detect/               ← 训练日志(gitignore)
```

---

## 🧠 YOLOv8 原理速览

> 详细数学公式与网络结构图见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**YOLO = You Only Look Once** —— 把目标检测当 end-to-end 回归问题, 一次 forward 出所有框.

### YOLOv8 相比 v5/v7 的关键改进

| 模块 | 改进点 |
|---|---|
| Backbone | C2f(替代 C3,梯度流更密集) |
| Head | **Anchor-free**(不再调 anchor 尺寸) |
| 标签分配 | **TaskAlignedAssigner**(对齐度 = 分类得分 × IoU^α) |
| Loss bbox | **CIoU + DFL**(Distribution Focal Loss) |
| Loss cls | **Varifocal Loss**(关注难分样本) |

### 训练 Loss

```
L = 7.5 · L_box(CIoU+DFL) + 0.5 · L_cls(Varifocal) + 1.5 · L_dfl
```

### 推理 Pipeline

```
原图 → letterbox 到 640 → 归一化 → forward → 8400 个预测
    → conf 阈值过滤 → NMS → 还原原图坐标
```

---

## 🏋️ 训练与数据集

### 数据集: Kaggle Hard Hat Workers

- **来源**: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
- **规模**: 5000 张图(416×416 PNG),25502 个标注框
- **类别**: `helmet`(18966)/`head`(5785)/`person`(751)
- **划分**: 4000 train / 500 val / 500 test

### 复现训练

```bash
# 1. 从 Kaggle 下载 archive.zip(1.3GB)放到 D:\archive.zip

# 2. 解压
mkdir -p datasets/HardHat
python -c "import zipfile; zipfile.ZipFile('D:/archive.zip').extractall('datasets/HardHat')"

# 3. VOC → YOLO
python src/prepare_hardhat.py

# 4. 训练(默认读 configs/train_params.yaml: yolov8s / 100ep / batch16 / imgsz640 / workers4)
python src/train.py
# 等价显式写法:
#   python src/train.py --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640 --workers 4
# 或快速试验: python src/train.py --epochs 20 --batch 8
```

### 关键训练参数(RTX 5060 8GB)

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `model` | yolov8s.pt | 精度/速度平衡(11.2M 参数); 求快用 yolov8n.pt |
| `batch` | 16 | yolov8s@640 稳跑; 显存紧降到 8 |
| `imgsz` | 640 | YOLOv8 默认; HardHat 原图 416 也可 |
| `half` (AMP) | True | 必须,显存减半速度 2x |
| `workers` | 4 | Windows 上 8 在 epoch 末尾关 Mosaic 时易 OOM |

### 实时监控训练

```bash
# CSV 数据流
tail -f runs/detect/hardhat_100ep/results.csv

# GPU 占用
nvidia-smi -l 1

# 训练曲线实时刷新(每 epoch 自动重画)
# 直接打开 runs/detect/hardhat_100ep/results.png
```

---

## ❓ 常见问题

### `CUDA error: no kernel image is available for execution on the device`

→ PyTorch 装错了,RTX 5060 必须 cu128:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### `CUDA out of memory`

按顺序试:
1. 加 `--half`
2. 降 `--batch 8`
3. 降 `--imgsz 416`
4. 关浏览器/QQ/Steam

### `torch.cuda.is_available()` 返回 False

→ 同上,重装 PyTorch cu128.

### 摄像头打不开

→ 加 `--cam 1`,或检查 Windows 隐私设置里的相机权限.

### 训练在 epoch 90+ MemoryError

→ `workers=4`(默认 8 在 Windows 上太激进). 已写入 `train_params.yaml`.

### 预训练权重 `yolov8s.pt` 下载失败 (SSL CERTIFICATE_VERIFY_FAILED)

国内访问 GitHub releases 会报证书吊销检查失败 (`CRYPT_E_NO_REVOCATION_CHECK`),
导致 `YOLO('yolov8s.pt')` 第一次自动下载就挂. 解法: 用镜像手动下到项目根目录,
ultralytics 会优先用本地文件不再联网:

```bash
curl -L -o yolov8s.pt https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt
# 备用镜像: 把 https://ghfast.top 换成 https://gh-proxy.com / https://mirror.ghproxy.com
```

### `conda create` 卡在 repodata / HTTP 000 连接失败

Anaconda 商业许可政策后, 国内 defaults 镜像(清华/北外/中科大)大多已撤下或 302
重定向到失效地址. 别死磕 conda 源, 两条出路:

1. **直接用 pip 建环境**(本项目推荐): 用任意 Python 3.11 跑
   `pip install torch --index-url https://download.pytorch.org/whl/cu128` 即可,
   完全绕开 conda defaults.
2. **复用已有 torch 环境**: 只要 `torch.cuda.is_available()` 为 True 且
   `torch.version.cuda == '12.8'`(支持 sm_120) 就能直接用, 不必新建 yolov8 环境.

更多问题见 **[USAGE.md §9](USAGE.md#9-常见问题排查)**.

---

## 📄 License

- **代码**: MIT
- **数据集 Kaggle Hard Hat Workers**: CC0 1.0(完全公有领域,商用无限制)
- **YOLOv8 / ultralytics**: AGPL-3.0(商用需购买 Ultralytics 商用 License)

---

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 实现
- [andrewmvd / Kaggle Hard Hat Workers](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) — 训练数据集
- [清华 TUNA](https://mirrors.tuna.tsinghua.edu.cn/) — 镜像源加速
