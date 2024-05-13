#!/bin/sh

# 检查是否存在名为venv的虚拟环境
if [ ! -d "venv" ]; then
  # 创建一个新的虚拟环境
  python3.10 -m venv venv

  # 安装依赖
  venv/bin/pip install -r requirements.txt
fi

# 激活虚拟环境并运行Python脚本
venv/bin/python3.10 Qt_ONNX_key.py