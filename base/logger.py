# xx_school_rag/base/logger.py
# 该脚本用于: 日志配置（支持控制台 + 文件双输出）

import logging
import os
from base.config import config  # 从 base.config 导入配置单例


# todo 1. 配置日志系统
def setup_logger(log_file=None, logger_name='xx_school_rag', level=logging.INFO):
    """
    创建并返回一个日志记录器，支持同时输出到控制台和文件。
    :param log_file: 日志文件路径，默认使用 config.LOG_FILE
    :param logger_name: 日志器名称，默认 'xx_school_rag'
    :param level: 日志级别，默认 INFO
    :return: 配置好的日志记录器
    """
    # 1. 确定日志文件路径
    if log_file is None:
        log_file = config.LOG_FILE  # 从配置获取

    # 2. 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 3. 创建日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # 4. 避免重复添加处理器
    if not logger.handlers:
        # 4.1 控制台处理器（输出到终端）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 4.2 文件处理器（输出到文件）
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)

        # 4.3 定义日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 4.4 添加处理器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # 5. 阻止日志向根日志器传播（避免重复输出）
    logger.propagate = False

    return logger


# todo 2. 创建全局日志器实例（供其他模块导入使用）
logger = setup_logger()


# todo 3. 测试代码
if __name__ == '__main__':
    logger.debug('这是一条 DEBUG 信息（不会显示，因为 level=INFO）')
    logger.info('这是一条 INFO 信息')
    logger.warning('这是一条 WARNING 信息')
    logger.error('这是一条 ERROR 信息')
    print(f"\n日志文件位置: {config.LOG_FILE}")