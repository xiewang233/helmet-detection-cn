# 📖 YOLOv8 安全帽检测 · 使用说明书

> 本说明书只讲**怎么用训练好的模型**。模型已就绪在 `models/hardhat_best.pt`。
>
> 训练相关内容请看 [README.md](README.md)。

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [5 秒快速上手](#2-5-秒快速上手)
3. [图片检测详解](#3-图片检测详解)
4. [视频检测详解](#4-视频检测详解)
5. [摄像头实时检测详解](#5-摄像头实时检测详解)
6. [批量检测](#6-批量检测)
7. [参数完全手册](#7-参数完全手册)
8. [结果文件说明](#8-结果文件说明)
9. [常见问题排查](#9-常见问题排查)
10. [高级用法](#10-高级用法)

---

## 1. 环境准备

### 1.1 首次使用前

确保已运行过 `install.bat`(详见 README.md §4)。如果还没装,先去装。

### 1.2 每次使用前(重要!)

**所有命令都必须先激活 conda 环境**,否则会报 `ModuleNotFoundError`。

#### 方式 1:双击激活(推荐新手)

```
双击 activate.bat
```

弹出窗口显示 `[OK] 已激活 yolov8 环境` 就成功了,**后续命令在这个窗口里输入**。

#### 方式 2:命令行激活

```bash
conda activate yolov8
```

激活后,命令行提示符前面会出现 `(yolov8)` 标识:

```
(yolov8) C:\Users\Administrator\Desktop\YOLOv8-Helmet-Detection>
```

### 1.3 验证环境

```bash
python src/check_gpu.py
```

看到 `[OK] GPU 完全就绪` 就 OK。

---

## 2. 5 秒快速上手

**最快路径:检测一张图片**。

```bash
# 1. 把你要检测的图片放到 data/images/ 目录,比如 test.jpg
# 2. 双击 activate.bat 激活环境
# 3. 运行:
python src/detect_image.py --model models/hardhat_best.pt --source data/images/test.jpg --save

# 4. 打开结果:
#    data/output/image_pred/test.jpg
```

**怎么读结果**:
- 🟢 **绿框** + `helmet 0.95` = 戴了安全帽(95% 置信度)
- 🔴 **红框** + `head 0.92` = 没戴安全帽的裸头

---

## 3. 图片检测详解

### 3.1 基本命令

```bash
python src/detect_image.py --model <权重> --source <图片> [选项]
```

### 3.2 常用例子

```bash
# 单张图片
python src/detect_image.py --model models/hardhat_best.pt --source D:/photos/site1.jpg --save

# 项目内图片
python src/detect_image.py --model models/hardhat_best.pt --source data/images/test.jpg --save

# 用 FP16 加速(RTX 5060 推荐,显存减半)
python src/detect_image.py --model models/hardhat_best.pt --source test.jpg --save --half

# 调高置信度阈值(只保留高置信度检测,减少误报)
python src/detect_image.py --model models/hardhat_best.pt --source test.jpg --save --conf 0.5

# 不弹窗,只保存(适合批量处理)
python src/detect_image.py --model models/hardhat_best.pt --source test.jpg --save --no-show
```

### 3.3 关键参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--model` | yolov8n.pt | 模型权重路径,本项目用 `models/hardhat_best.pt` |
| `--source` | (必填) | 图片路径 |
| `--conf` | 0.25 | 置信度阈值,范围 0-1。**调高减少误报,调低增加召回** |
| `--iou` | 0.7 | NMS 重叠阈值。**调低更激进合并重叠框** |
| `--imgsz` | 640 | 输入分辨率。**降低到 416 显存占用减半,精度略降** |
| `--half` | False | FP16 推理。**RTX 5060 必开,显存减半** |
| `--save` | False | 保存到 `data/output/image_pred/` |
| `--no-show` | False | 不弹窗显示 |

### 3.4 终端输出怎么读

```
模型: models/hardhat_best.pt
输入: D:/photos/site1.jpg
设备: cuda (RTX 5060)

  site1.jpg: 检测到 8 个 -> ['helmet', 'helmet', 'head', 'helmet', ...]

共 1 张图片, 总检测 8 个目标.
结果已保存到: data/output/image_pred
```

---

## 4. 视频检测详解

### 4.1 基本命令

```bash
python src/detect_video.py --model <权重> --source <视频> [选项]
```

### 4.2 常用例子

```bash
# 处理一个 mp4
python src/detect_video.py --model models/hardhat_best.pt --source D:/videos/site.mp4 --save

# 不弹窗,后台处理(适合长视频)
python src/detect_video.py --model models/hardhat_best.pt --source site.mp4 --save --no-show

# FP16 加速
python src/detect_video.py --model models/hardhat_best.pt --source site.mp4 --save --half
```

### 4.3 输出

- **标注后的视频**:`data/output/video_pred/site.avi`
- **终端实时进度**:每 30 帧打印一次累计检测数

### 4.4 性能预期(RTX 5060)

| 视频分辨率 | FPS | 1 分钟视频处理时间 |
|---|---|---|
| 720p | ~80 | 0.75 秒 |
| 1080p | ~50 | 1.2 秒 |
| 4K | ~15 | 4 秒 |

---

## 5. 摄像头实时检测详解

### 5.1 基本命令

```bash
python src/detect_camera.py --model <权重> [选项]
```

### 5.2 常用例子

```bash
# 用默认摄像头(笔记本自带)
python src/detect_camera.py --model models/hardhat_best.pt

# FP16 加速
python src/detect_camera.py --model models/hardhat_best.pt --half

# 用外接 USB 摄像头
python src/detect_camera.py --model models/hardhat_best.pt --cam 1

# 指定分辨率
python src/detect_camera.py --model models/hardhat_best.pt --width 1920 --height 1080
```

### 5.3 操作快捷键

| 键 | 作用 |
|---|---|
| `q` | 退出 |
| `s` | 截图当前帧(保存到 `data/output/`) |

### 5.4 窗口显示

```
┌──────────────────────────────────────┐
│ FPS: 75.3                            │ ← 实时 FPS
│                                      │
│   🟢 [helmet 0.95]                   │ ← 绿框 = 戴帽
│       │                              │
│       ▼                              │
│   👤 person                           │
│                                      │
│   🔴 [head 0.88]                     │ ← 红框 = 没戴
│                                      │
└──────────────────────────────────────┘
```

---

## 6. 批量检测

### 6.1 批量图片

把所有图片放进一个目录,然后:

```bash
python src/detect_image.py --model models/hardhat_best.pt --source data/images/ --save --no-show
```

支持 `.jpg` / `.png` / `.jpeg`,所有结果都会保存到 `data/output/image_pred/`。

### 6.2 用 Python 脚本批量处理

```python
# my_batch.py
from ultralytics import YOLO
from pathlib import Path

model = YOLO('models/hardhat_best.pt')
imgs = list(Path('D:/photos').glob('*.jpg'))

results = model.predict(
    source=[str(p) for p in imgs],
    conf=0.25,
    device=0,
    save=True,
    project='data/output',
    name='batch_run',
    exist_ok=True,
)

# 统计
total_helmet = 0
total_head = 0
for r in results:
    for cls in r.boxes.cls:
        if int(cls) == 0: total_helmet += 1  # helmet
        elif int(cls) == 1: total_head += 1  # head

print(f"共 {len(results)} 张图")
print(f"  戴帽: {total_helmet}")
print(f"  没戴: {total_head}")
```

运行:

```bash
python my_batch.py
```

---

## 7. 参数完全手册

### 7.1 `detect_image.py`

```
参数              默认值      说明
--model          yolov8n.pt  模型权重路径
--source         (必填)      图片路径或目录
--conf           0.25        置信度阈值 (0-1)
--iou            0.7         NMS IoU 阈值 (0-1)
--imgsz          640         输入分辨率 (320/416/640/1280)
--half           False       FP16 推理(显存减半)
--save           False       保存结果到 data/output/
--no-show        False       不弹窗
```

### 7.2 `detect_video.py`

```
参数              默认值      说明
--model          yolov8n.pt  模型权重
--source         (必填)      视频文件路径
--conf           0.25        置信度阈值
--iou            0.7         NMS IoU
--imgsz          640         输入分辨率
--half           False       FP16
--save           False       保存标注后视频
--no-show        False       不弹窗播放
```

### 7.3 `detect_camera.py`

```
参数              默认值      说明
--model          yolov8n.pt  模型权重
--cam            0           摄像头索引 (0=内置, 1=外接 USB)
--conf           0.25        置信度阈值
--iou            0.7         NMS IoU
--imgsz          640         输入分辨率
--half           False       FP16
--width          1280        请求的摄像头分辨率宽
--height         720         请求的摄像头分辨率高
```

### 7.4 通用建议

| 场景 | 推荐参数 |
|---|---|
| 追求速度 | `--half --imgsz 416` |
| 追求精度 | `--conf 0.4 --imgsz 640` |
| 显存不够 | `--half --imgsz 416` |
| 减少误报 | `--conf 0.5`(只保留置信度 50% 以上的) |
| 减少漏检 | `--conf 0.15`(让低置信度也显示) |

---

## 8. 结果文件说明

### 8.1 图片检测

```
data/output/image_pred/
└── 你的图片.jpg       ← 带检测框的图片
```

### 8.2 视频检测

```
data/output/video_pred/
└── 你的视频.avi       ← 带检测框的视频
```

### 8.3 摄像头截图

```
data/output/
└── snapshot_1718354123.jpg   ← 按 s 截图时生成
```

### 8.4 检测框颜色

| 颜色 | 类别 | 含义 |
|---|---|---|
| 🟢 绿色 | `helmet` | 戴了安全帽 |
| 🔴 红色 | `head` | 没戴安全帽的裸头 |
| 🟡 黄色 | `person` | 整个人(本项目很少触发) |

> 注:实际颜色由 ultralytics 默认配色决定,绿/红是 helmet/head 最常见的对应。

### 8.5 框上的文字格式

```
helmet 0.95
└┬───┘ └┬┘
 │      └─ 置信度(0-1)
 └─ 类别名
```

---

## 9. 常见问题排查

### 9.1 启动问题

#### ❌ `ModuleNotFoundError: No module named 'ultralytics'`

**原因**:没有激活 yolov8 环境。

**解决**:
```bash
conda activate yolov8
# 或双击 activate.bat
```

#### ❌ `CUDA error: no kernel image is available for execution on the device`

**原因**:PyTorch 装成了 CPU 版,或者版本不支持 RTX 5060 Blackwell。

**解决**:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### ❌ `torch.cuda.is_available()` 返回 False

同上,重装 PyTorch cu128 版。

### 9.2 运行时问题

#### ❌ `CUDA out of memory`

**原因**:显存不够。

**按顺序试**:
```bash
# 方案 1:开 FP16
python src/detect_image.py ... --half

# 方案 2:降分辨率
python src/detect_image.py ... --imgsz 416

# 方案 3:关掉其他占显存的程序
# 浏览器、QQ、Steam、其他 Python 进程
```

#### ❌ 摄像头打不开 / 黑屏

**解决**:
```bash
# 试别的索引
python src/detect_camera.py --model models/hardhat_best.pt --cam 1

# 检查权限:设置 → 隐私 → 相机 → 允许应用访问
```

#### ❌ 中文乱码

终端运行:
```bash
chcp 65001
```

#### ❌ 找不到 `models/hardhat_best.pt`

**原因**:权重文件没生成或被删。

**解决**:
```bash
# 看下还在不在
ls models/

# 如果不在,从训练产物复制
cp runs/detect/runs/detect/hardhat_100ep/weights/best.pt models/hardhat_best.pt
```

### 9.3 检测效果问题

#### ❌ 漏检很多(模型识别不出)

**调低置信度阈值**:
```bash
python src/detect_image.py ... --conf 0.15
```

#### ❌ 误报很多(背景被识别成 helmet)

**调高置信度阈值**:
```bash
python src/detect_image.py ... --conf 0.5
```

#### ❌ 重叠框太多

**调低 IoU 阈值**:
```bash
python src/detect_image.py ... --iou 0.5
```

### 9.4 性能问题

#### ❌ FPS 很低(< 30)

**按顺序试**:
1. 开 FP16:加 `--half`
2. 关掉其他占 GPU 的程序
3. 检查 GPU 占用:`nvidia-smi`(看是不是被别的进程占了)
4. 降低分辨率:加 `--imgsz 416`

#### ❌ CPU 占用 100%,GPU 没动

**原因**:PyTorch 装成了 CPU 版,GPU 没在跑。

**解决**:同 9.1 的 `no kernel image` 问题。

---

## 10. 高级用法

### 10.1 用 Python 调用模型(编程方式)

```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('models/hardhat_best.pt')

# 推理一张图
results = model.predict(
    source='test.jpg',
    conf=0.25,
    device=0,        # GPU
    half=True,       # FP16
    save=False,
)

# 解析结果
for r in results:
    for box in r.boxes:
        cls = int(box.cls)              # 类别 ID (0=helmet, 1=head)
        conf = float(box.conf)          # 置信度
        x1, y1, x2, y2 = box.xyxy[0]    # 边界框坐标
        name = r.names[cls]             # 类别名

        print(f"{name} {conf:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

### 10.2 接入 RTSP 网络摄像头

把 `--source` 换成 RTSP URL:

```bash
python src/detect_video.py --model models/hardhat_best.pt \
    --source rtsp://admin:password@192.168.1.100:554/stream
```

### 10.3 用 ONNX Runtime 部署(不依赖 PyTorch)

```bash
# 先导出
python src/export_model.py --weights models/hardhat_best.pt --format onnx --simplify

# 然后用 onnxruntime 推理(跨平台,无 PyTorch 依赖)
# 详见 https://docs.ultralytics.com/guides/azure-ml/
```

### 10.4 集成到 Web 服务

```python
# server.py
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO

app = FastAPI()
model = YOLO('models/hardhat_best.pt')

@app.post("/detect")
async def detect(file: UploadFile):
    img = await file.read()
    results = model.predict(img, device=0, verbose=False)
    return results[0].to_json()

# 启动: uvicorn server:app --port 8000
```

测试:
```bash
curl -X POST -F "file=@test.jpg" http://localhost:8000/detect
```

### 10.5 触发告警(没戴帽就报警)

```python
# alarm.py
from ultralytics import YOLO
import cv2

model = YOLO('models/hardhat_best.pt')
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break

    results = model.predict(frame, conf=0.4, device=0, verbose=False)
    r = results[0]

    # 检查是否有没戴帽的人
    unsafe = sum(1 for c in r.boxes.cls if int(c) == 1)
    if unsafe > 0:
        print(f"⚠️  发现 {unsafe} 个未戴安全帽人员!")

    cv2.imshow('monitor', r.plot())
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
```

---

## 📞 命令速查卡(打印用)

```
┌─────────────────────────────────────────────────────────────────┐
│  YOLOv8 安全帽检测 · 命令速查                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  激活环境:   conda activate yolov8                              │
│             或双击 activate.bat                                 │
│                                                                 │
│  图片检测:   python src/detect_image.py \                       │
│               --model models/hardhat_best.pt \                  │
│               --source 图片.jpg --save --half                   │
│                                                                 │
│  视频检测:   python src/detect_video.py \                       │
│               --model models/hardhat_best.pt \                  │
│               --source 视频.mp4 --save                          │
│                                                                 │
│  摄像头:     python src/detect_camera.py \                      │
│               --model models/hardhat_best.pt (按 q 退出)         │
│                                                                 │
│  性能压测:   python src/benchmark.py                            │
│                                                                 │
│  导出 ONNX:  python src/export_model.py \                       │
│               --weights models/hardhat_best.pt --format onnx    │
│                                                                 │
│  结果位置:   data/output/                                       │
│             绿框=helmet, 红框=head                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🆘 需要更多帮助

- 📖 项目原理与训练:看 [README.md](README.md)
- 🔧 网络结构详解:看 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 🐛 提 issue:https://github.com/xiewang233/helmet-detection-cn/issues
- 📚 ultralytics 官方文档:https://docs.ultralytics.com
