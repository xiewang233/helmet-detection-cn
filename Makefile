# YOLOv8 安全帽检测 - 常用命令入口
#
# 用法 (Git Bash / 装了 make 的环境):
#   make check          # GPU/CUDA 自检
#   make prepare        # HardHat VOC -> YOLO
#   make train          # 训练 (读 configs/train_params.yaml)
#   make detect IMG=xxx.jpg MODEL=models/hardhat_best.pt
#   make benchmark      # 性能压测 (落盘 CSV+PNG)
#   make export-onnx W=models/hardhat_best.pt
#   make clean          # 清本次训练产物
#
# 没装 make 也没关系, 每个 target 下一行就是等效命令。
# 先激活环境: conda activate pytorch (或你装 torch 的那个环境)

PY ?= python

.PHONY: check prepare train detect benchmark export-onnx export-engine clean help

help:  ## 列出所有 target
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

check:  ## GPU/CUDA 自检
	$(PY) src/check_gpu.py

prepare:  ## HardHat VOC -> YOLO 格式 (8:1:1)
	$(PY) src/prepare_hardhat.py

train:  ## 训练 (读 configs/train_params.yaml, 默认 yolov8s 100ep)
	$(PY) src/train.py

detect:  ## 图片推理: make detect IMG=x.jpg MODEL=models/hardhat_best.pt
	$(PY) src/detect_image.py --model $(MODEL) --source $(IMG) --save --no-show

benchmark:  ## 性能压测, 结果存 runs/benchmark/
	$(PY) src/benchmark.py --out runs/benchmark

export-onnx:  ## 导 ONNX: make export-onnx W=models/hardhat_best.pt
	$(PY) src/export_model.py --weights $(W) --format onnx --simplify

export-engine:  ## 导 TensorRT FP16: make export-engine W=models/hardhat_best.pt
	$(PY) src/export_model.py --weights $(W) --format engine --half

clean:  ## 清理 runs/detect/hardhat_100ep
	rm -rf runs/detect/hardhat_100ep
