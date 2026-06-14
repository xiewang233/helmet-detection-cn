@echo off
chcp 65001 >nul
REM ============================================
REM   激活 yolov8 conda 环境的快捷入口
REM ============================================
set "CONDA_BASE=C:\ProgramData\miniconda3"
call "%CONDA_BASE%\Scripts\activate.bat" yolov8
echo.
echo [OK] 已激活 yolov8 环境
echo Python 解释器: %CONDA_PREFIX%\python.exe
python --version
echo.
echo 可用命令:
echo   python src\check_gpu.py        REM 检查 GPU 状态
echo   python src\detect_image.py --help
echo   python src\detect_video.py --help
echo   python src\detect_camera.py --help
echo   python src\benchmark.py
echo.
