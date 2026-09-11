# xx_school_rag/core/bm25_search.py
# 该脚本用于: 迎新问答「快速通道」——精确缓存 + BM25 近似匹配 + MySQL 取答案。
#
# 返回值统一为 (answer, need_rag):
#   (None,   True)  -> 快速通道未解决，交给后续 RAG 系统
#   (answer, False) -> 已命中，直接把答案返回给学生


from core.bm25_matcher import best_match, tokenize  # 分词函数与 BM25 匹配函数
from core.mysql_client import MySQLClient  # MySQL 客户端，用于按问题查答案、拉取全量问题
from core.redis_client import RedisClient  # Redis 客户端，用于缓存问题库和精确答案缓存
from base import Config, logger  # 项目统一配置类和日志对象

_KEY_ORIGINAL = "qa_original_questions"  # Redis 中存放"原始问题列表"的键名
_KEY_TOKENIZED = "qa_tokenized_questions"  # Redis 中存放"分词后问题列表"的键名
_MAX_QUERY_LEN = 200  # 允许的最大查询长度，超出视为无效查询


class BM25Search:
    # todo 1. 初始化：客户端 + 加载问题库
    def __init__(self, redis_client, mysql_client):
        self.logger = logger  # 保存日志对象为实例属性
        self.redis_client = redis_client  # 保存传入的 Redis 客户端实例
        self.mysql_client = mysql_client  # 保存传入的 MySQL 客户端实例
        # 分词后的问题列表 list[list[str]]，与 original_questions 下标对齐
        self.questions = None  # 初始为空，稍后由 _load_data 填充
        # 原始问题列表 list[str]
        self.original_questions = None  # 初始为空，稍后由 _load_data 填充
        self._load_data()  # 构造时立即加载问题库数据

    # todo 2. 数据加载：优先 Redis，缺失则从 MySQL 重建并回写
    def _load_data(self):
        self.original_questions = self.redis_client.get_data(_KEY_ORIGINAL)  # 尝试从 Redis 读取原始问题列表缓存
        tokenized = self.redis_client.get_data(_KEY_TOKENIZED)  # 尝试从 Redis 读取分词后问题列表缓存

        if not self.original_questions or not tokenized:  # 只要任一缓存缺失，就需要重新从数据库构建
            rows = self.mysql_client.fetch_questions()   # [(问题,), (问题,), ...]  从 MySQL 拉取全量问题
            if not rows:  # 数据库里也没有问题数据
                self.logger.warning("未从 MySQL 加载到任何问题数据，请先导入 qa_knowledge")  # 记录警告日志
                self.original_questions, self.questions = [], []  # 置为空列表，避免后续访问 None 报错
                return  # 提前结束，不再回写缓存
            self.original_questions = [r[0] for r in rows]  # 从查询结果的元组中取出每行第一个字段（问题文本）
            tokenized = [tokenize(q) for q in self.original_questions]  # 对每个原始问题分词
            self.redis_client.set_data(_KEY_ORIGINAL, self.original_questions)  # 将原始问题列表写回 Redis 缓存
            self.redis_client.set_data(_KEY_TOKENIZED, tokenized)  # 将分词后问题列表写回 Redis 缓存

        self.questions = tokenized  # 将分词结果赋值给实例属性，供后续搜索使用
        self.logger.info(f"BM25 问题库加载完成：{len(self.questions)} 条")  # 记录加载完成的问题条数

    # todo 3. 核心搜索
    def search(self, query, threshold=None):
        """
        :return: (answer, False) 命中 / (None, True) 未命中需走 RAG
        """
        if threshold is None:  # 未传入阈值时使用配置中的默认阈值
            threshold = Config().THRESHOLD

        # 3.1 查询有效性
        if not query or not isinstance(query, str) or len(query.strip()) > _MAX_QUERY_LEN:  # 空值/非字符串/超长查询均视为无效
            self.logger.info(f"无效查询: {query!r}")  # 记录无效查询日志
            return None, True  # 直接转交给 RAG 处理
        query = query.strip()  # 去除首尾空白，规范化查询文本

        # 3.2 Redis 精确缓存
        cached = self.redis_client.get_answer(query)  # 尝试按原文精确匹配缓存的答案
        if cached:  # 命中精确缓存
            self.logger.info(f"缓存命中: {query}")  # 记录命中日志
            return cached, False  # 直接返回缓存答案，无需走 RAG

        # 3.3 分词
        query_tokens = tokenize(query)  # 对查询文本分词
        if not query_tokens or not self.questions:  # 分词结果为空 或 问题库为空，都无法继续匹配
            self.logger.info("分词为空或问题库为空，转 RAG")  # 记录日志
            return None, True  # 转交给 RAG 处理

        # 3.4 BM25 近似匹配
        try:
            best_idx, score = best_match(query_tokens, self.questions)  # 在问题库中找出与查询最相似的一条及其归一化相似度
        except Exception as e:  # 捕获计算过程中可能出现的任何异常
            self.logger.error(f"BM25 计算异常: {e}")  # 记录错误日志
            return None, True  # 出错时保守地转交给 RAG 处理

        # 3.5 阈值判断
        if best_idx < 0 or score < threshold:  # 未找到匹配 或 相似度低于阈值，说明不够可靠
            self.logger.info(f"未找到可靠答案，最高相似度 {score:.3f} < 阈值 {threshold:.3f}，转 RAG")  # 记录日志
            return None, True  # 转交给 RAG 处理

        # 3.6 取原始问题 -> MySQL 查答案
        original_question = self.original_questions[best_idx]  # 根据下标取出对应的原始问题文本
        answer = self.mysql_client.fetch_answer(original_question)  # 用原始问题去 MySQL 中查询对应答案
        if not answer:  # 数据库里未查到该问题对应的答案
            self.logger.info(f"MySQL 未查到答案: {original_question}，转 RAG")  # 记录日志
            return None, True  # 转交给 RAG 处理

        # 3.7 回写缓存并返回
        self.redis_client.set_answer(query, answer)  # 将本次查询与答案写入精确缓存，加速下次同样的提问
        self.logger.info(f"BM25 命中: {query!r} ~= {original_question!r} (相似度 {score:.3f})")  # 记录命中详情日志
        return answer, False  # 返回命中的答案，无需再走 RAG


if __name__ == '__main__':  # 直接运行本文件时执行的自测代码
    bm25 = BM25Search(RedisClient(), MySQLClient())  # 实例化 BM25Search，注入真实的 Redis / MySQL 客户端
    for q in ["大一新生去哪里报到？", "图书馆的开放时间", "怎么申请助学贷款", ""]:  # 遍历若干测试查询（含一个空字符串）
        print(repr(q), "->", bm25.search(q))  # 打印每个查询的搜索结果
