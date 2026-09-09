# EduRAG_Project/core/redis_client.py
# 实现Redis数据库连接及数据操作

# 导包
import redis
import json
from base import Config, logger

# todo 1. 定义Redis客户端类 -> 封装Redis连接, 数据存储, 数据获取, 答案查询等功能.
class RedisClient:
    # todo 1.1 初始化方法, 获取Redis连接, 初始化日志, 处理连接异常.
    def __init__(self):
        self.logger = logger
        # 获取Redis连接
        try:
            self.redis_client = redis.Redis(host=Config().REDIS_HOST, port=Config().REDIS_PORT, password=Config().REDIS_PASSWORD, db=Config().REDIS_DB,decode_responses=True)
        except redis.RedisError as e:
            self.logger.error(f"Redis连接异常: {e}")
            raise
        # 记录日志
        self.logger.info(f"Redis连接成功: {Config().REDIS_HOST}:{Config().REDIS_PORT}, db={Config().REDIS_DB}")

    # todo 1.2 存储数据方法, 存储数据到Redis数据库.
    def set_data(self, key, value):
        try:
            self.redis_client.set(key, json.dumps(value, ensure_ascii=False)) # ensure_ascii: 保证中文等非ASCII字符正常显式.
            self.logger.info(f"redis数据存储成功: {key}")
        except redis.RedisError as e:
            self.logger.error(f"Redis存储数据异常: {e}")
            raise

    # todo 1.3 获取数据方法, 获取Redis数据库中的数据.
    def get_data(self, key):
        try:
            data = self.redis_client.get(key)
            self.logger.info(f'Redis获取数据成功: {key}')
            return json.loads(data) if data else None
        except redis.RedisError as e:
            logger.error(f"Redis获取数据异常: {e}")
            return None

    # todo 1.4 根据查询内容从Redis获取缓存的答案 -> 键格式固定为: "answer:{query}"
    def get_answer(self, query):
        try:
            # 1. 构建键名, 区分不同类型的缓存数据.
            answer = self.redis_client.get(f"answer:{query}")
            # 2. 若存在缓存答案, 记录INFO日志, 并返回内容即可.
            if answer:
                self.logger.info(f'从Redis中获取答案成功: {query}')
                return answer
        except redis.RedisError as e:
            self.logger.error(f"Redis获取答案异常: {e}")
            return None

    # todo 1.5 热点问题存储
    def set_answer(self, query: str, answer: str):
        try:
            key = f"answer:{query}"
            self.redis_client.set(key, answer)
            self.logger.info(f"热点问答缓存成功: {query[:20]}...")
        except redis.RedisError as e:
            self.logger.error(f"热点问答缓存失败: {e}")



if __name__ == '__main__':
    redis_client = RedisClient()
