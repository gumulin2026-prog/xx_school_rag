# EduRAG_Project/core/redis_client.py
# 实现Redis数据库连接及数据操作

# 导包
import redis  # 导入 redis 官方客户端库，用于连接和操作 Redis 数据库
import json   # 导入 json 模块，用于对象与 JSON 字符串之间的序列化/反序列化
from base import Config, logger  # 导入项目统一的配置类和日志对象

# todo 1. 定义Redis客户端类 -> 封装Redis连接, 数据存储, 数据获取, 答案查询等功能.
class RedisClient:
    # todo 1.1 初始化方法, 获取Redis连接, 初始化日志, 处理连接异常.
    def __init__(self):
        self.logger = logger  # 将全局日志对象保存为实例属性，方便类内各方法调用
        # 获取Redis连接
        try:
            # 使用配置中的主机、端口、密码、库编号创建 Redis 连接；decode_responses=True 表示自动将字节解码为字符串
            self.redis_client = redis.Redis(host=Config().REDIS_HOST, port=Config().REDIS_PORT, password=Config().REDIS_PASSWORD, db=Config().REDIS_DB,decode_responses=True)
        except redis.RedisError as e:  # 捕获连接过程中可能出现的 Redis 异常
            self.logger.error(f"Redis连接异常: {e}")  # 记录错误日志
            raise  # 将异常继续向上抛出，交由调用方处理
        # 记录日志
        self.logger.info(f"Redis连接成功: {Config().REDIS_HOST}:{Config().REDIS_PORT}, db={Config().REDIS_DB}")  # 连接成功后记录一条信息日志

    # todo 1.2 存储数据方法, 存储数据到Redis数据库.
    def set_data(self, key, value):
        try:
            self.redis_client.set(key, json.dumps(value, ensure_ascii=False)) # ensure_ascii: 保证中文等非ASCII字符正常显式.
            self.logger.info(f"redis数据存储成功: {key}")  # 存储成功后记录日志
        except redis.RedisError as e:  # 捕获存储过程中可能出现的异常
            self.logger.error(f"Redis存储数据异常: {e}")  # 记录错误日志
            raise  # 继续向上抛出异常

    # todo 1.3 获取数据方法, 获取Redis数据库中的数据.
    def get_data(self, key):
        try:
            data = self.redis_client.get(key)  # 根据键从 Redis 中取出原始字符串数据（可能为 None）
            self.logger.info(f'Redis获取数据成功: {key}')  # 记录获取成功的日志
            return json.loads(data) if data else None  # 若数据存在则反序列化为 Python 对象，否则返回 None
        except redis.RedisError as e:  # 捕获获取过程中可能出现的异常
            logger.error(f"Redis获取数据异常: {e}")  # 记录错误日志
            return None  # 出现异常时返回 None，避免程序崩溃

    # todo 1.4 根据查询内容从Redis获取缓存的答案 -> 键格式固定为: "answer:{query}"
    def get_answer(self, query):
        try:
            # 1. 构建键名, 区分不同类型的缓存数据.
            answer = self.redis_client.get(f"answer:{query}")  # 按固定前缀拼接键名并查询缓存的答案
            # 2. 若存在缓存答案, 记录INFO日志, 并返回内容即可.
            if answer:  # 判断是否命中缓存
                self.logger.info(f'从Redis中获取答案成功: {query}')  # 命中时记录日志
                return answer  # 返回缓存的答案文本
        except redis.RedisError as e:  # 捕获查询过程中可能出现的异常
            self.logger.error(f"Redis获取答案异常: {e}")  # 记录错误日志
            return None  # 异常时返回 None（未命中缓存时函数隐式返回 None）

    # todo 1.5 热点问题存储
    def set_answer(self, query: str, answer: str):
        try:
            key = f"answer:{query}"  # 按固定前缀拼接生成缓存键
            self.redis_client.set(key, answer)  # 将答案文本以字符串形式写入 Redis
            self.logger.info(f"热点问答缓存成功: {query[:20]}...")  # 记录成功日志（只截取问题前20个字符，避免日志过长）
        except redis.RedisError as e:  # 捕获写入过程中可能出现的异常
            self.logger.error(f"热点问答缓存失败: {e}")  # 记录错误日志（此处未抛出异常，避免影响主流程）



if __name__ == '__main__':  # 当该脚本被直接运行时执行，作为简单的连接自测
    redis_client = RedisClient()  # 实例化客户端，触发连接与日志输出
