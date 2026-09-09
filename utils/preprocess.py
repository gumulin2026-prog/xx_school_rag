# EduRAG_Project/utils/preprocess.py
# 用于文本处理: 切分文本

# 导包
import jieba
from base import logger

# todo 1.定义文本预处理函数.
def preprocess_text(text):
    # 预处理文本
    logger.info("开始预处理文本")
    try:
        # 分词并转换为小写
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []

# todo 2.程序的主入口.
if __name__ == '__main__':
    # 1. 调用preprocess_text()函数, 测试.
    print(preprocess_text("黑马程序员"))