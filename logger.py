"""
Z.R.I.C 引擎 — 统一日志模块 (logger.py)
所有模块通过 from logger import log 获取统一的 logger 实例。
日志同时输出到控制台和文件。
"""

import logging
import os
import sys

# 识别运行目录（与 main.py 保持一致）
if getattr(sys, 'frozen', False):
    _BASE_DIR = os.path.dirname(sys.executable)
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(_BASE_DIR, "Z.R.I.C.log")

# ---------------------------------------------------------
# 日志格式与等级
# ---------------------------------------------------------
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

# 控制台 handler（INFO 及以上）
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)

# 文件 handler（DEBUG 及以上，按 5MB 滚动）
try:
    from logging.handlers import RotatingFileHandler
    _file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_formatter)
except Exception:
    _file_handler = None


def get_logger(name: str = "Z.R.I.C") -> logging.Logger:
    """获取一个已配置好的 logger 实例。每个模块传入自己的名字即可。"""
    logger = logging.getLogger(name)
    if not logger.handlers:  # 避免重复添加 handler
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_console_handler)
        if _file_handler:
            logger.addHandler(_file_handler)
    return logger


# 默认 logger（快捷引用）
log = get_logger("Z.R.I.C")
