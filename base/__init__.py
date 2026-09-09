# xx_school_rag/base/__init__.py

# 导包
import os, sys

# 1. 获取当前文件夹所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取项目根目录的绝对路径
project_root = os.path.dirname(current_dir)  # 项目根目录 .../xx_shcool_rag

# 3. 将该模块目录添加到系统路径中
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 4. 添加项目根目录到系统路径中
if project_root not in sys.path:
    sys.path.insert(0,project_root)

# 5. 打印系统目录
# print(sys.path)

# 6. 导入自定义模块
from .config import Config
from .logger import logger   # ← 修改这里，加上点号
