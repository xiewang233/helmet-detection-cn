@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title YOLOv8 安全帽检测 - 一键安装

REM ============================================
REM  YOLOv8 安全帽检测 - 一键安装脚本
REM  适配: Windows 11 + NVIDIA RTX 5060 (Blackwell)
REM  Miniconda: C:\ProgramData\miniconda3
REM ============================================

set "CONDA_BASE=C:\ProgramData\miniconda3"
set "ENV_NAME=yolov8"
set "PY=%CONDA_BASE%\envs\%ENV_NAME%\python.exe"

echo.
echo ============================================
echo   YOLOv8 安全帽检测 - 一键安装
echo ============================================
echo.

REM ---- 检查 Miniconda ----
if not exist "%CONDA_BASE%\Scripts\conda.exe" (
    echo [X] 未找到 Miniconda: %CONDA_BASE%
    echo     请先下载安装: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)
echo [OK] Miniconda 已安装: %CONDA_BASE%

REM ---- 创建 conda 环境 ----
echo.
echo [1/4] 创建 conda 环境 %ENV_NAME% (Python 3.11) ...
if exist "%PY%" (
    echo      已存在, 跳过
) else (
    call "%CONDA_BASE%\Scripts\conda.exe" create -n %ENV_NAME% python=3.11 -y
    if errorlevel 1 (
        echo [X] conda 环境创建失败
        pause
        exit /b 1
    )
)

REM ---- 激活环境 ----
call "%CONDA_BASE%\Scripts\activate.bat" %ENV_NAME%

REM ---- 升级 pip ----
echo.
echo [2/4] 升级 pip ...
python -m pip install --upgrade pip

REM ---- 安装 PyTorch cu128 (关键!RTX 5060 Blackwell 必须) ----
echo.
echo [3/4] 安装 PyTorch cu128 (RTX 5060 必需, 约 2.5GB, 请耐心等待) ...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [X] PyTorch 安装失败, 请检查网络
    pause
    exit /b 1
)

REM ---- 安装 ultralytics 等依赖 (清华源) ----
echo.
echo [4/4] 安装 ultralytics 及其他依赖 (清华源) ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [X] ultralytics 安装失败
    pause
    exit /b 1
)

REM ---- 验证 GPU ----
echo.
echo ============================================
echo   验证 GPU 状态
echo ============================================
python src\check_gpu.py

echo.
echo ============================================
echo   安装完成!
echo ============================================
echo 后续使用前请先激活环境:
echo   activate.bat
echo 或:
echo   conda activate yolov8
echo.
pause
