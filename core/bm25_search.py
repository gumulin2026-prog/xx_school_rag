# xx_school_rag/core/bm25_search.py
# 该脚本用于: 迎新问答「快速通道」——精确缓存 + BM25 近似匹配 + MySQL 取答案。
#
# 返回值统一为 (answer, need_rag):
#   (None,   True)  -> 快速通道未解决，交给后续 RAG 系统
#   (answer, False) -> 已命中，直接把答案返回给学生
#
# 与 EduRAG 原版的区别：
#   - 分词走 core.bm25_matcher（jieba 领域词典 + 停用词）
#   - 相似度归一化用「追加文档法」(见 bm25_matcher.best_match)，取代原来的 softmax
#   - 命中时统一返回 (answer, False)，修正原版 (answer, True) 与文档不符的问题

from core.bm25_matcher import best_match, tokenize
from core.mysql_client import MySQLClient
from core.redis_client import RedisClient
from base import Config, logger

_KEY_ORIGINAL = "qa_original_questions"
_KEY_TOKENIZED = "qa_tokenized_questions"
_MAX_QUERY_LEN = 200


class BM25Search:
    # todo 1. 初始化：客户端 + 加载问题库
    def __init__(self, redis_client, mysql_client):
        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        # 分词后的问题列表 list[list[str]]，与 original_questions 下标对齐
        self.questions = None
        # 原始问题列表 list[str]
        self.original_questions = None
        self._load_data()

    # todo 2. 数据加载：优先 Redis，缺失则从 MySQL 重建并回写
    def _load_data(self):
        self.original_questions = self.redis_client.get_data(_KEY_ORIGINAL)
        tokenized = self.redis_client.get_data(_KEY_TOKENIZED)

        if not self.original_questions or not tokenized:
            rows = self.mysql_client.fetch_questions()   # [(问题,), (问题,), ...]
            if not rows:
                self.logger.warning("未从 MySQL 加载到任何问题数据，请先导入 qa_knowledge")
                self.original_questions, self.questions = [], []
                return
            self.original_questions = [r[0] for r in rows]
            tokenized = [tokenize(q) for q in self.original_questions]
            self.redis_client.set_data(_KEY_ORIGINAL, self.original_questions)
            self.redis_client.set_data(_KEY_TOKENIZED, tokenized)

        self.questions = tokenized
        self.logger.info(f"BM25 问题库加载完成：{len(self.questions)} 条")

    # todo 3. 核心搜索
    def search(self, query, threshold=None):
        """
        :return: (answer, False) 命中 / (None, True) 未命中需走 RAG
        """
        if threshold is None:
            threshold = Config().THRESHOLD

        # 3.1 查询有效性
        if not query or not isinstance(query, str) or len(query.strip()) > _MAX_QUERY_LEN:
            self.logger.info(f"无效查询: {query!r}")
            return None, True
        query = query.strip()

        # 3.2 Redis 精确缓存
        cached = self.redis_client.get_answer(query)
        if cached:
            self.logger.info(f"缓存命中: {query}")
            return cached, False

        # 3.3 分词
        query_tokens = tokenize(query)
        if not query_tokens or not self.questions:
            self.logger.info("分词为空或问题库为空，转 RAG")
            return None, True

        # 3.4 BM25 近似匹配
        try:
            best_idx, score = best_match(query_tokens, self.questions)
        except Exception as e:
            self.logger.error(f"BM25 计算异常: {e}")
            return None, True

        # 3.5 阈值判断
        if best_idx < 0 or score < threshold:
            self.logger.info(f"未找到可靠答案，最高相似度 {score:.3f} < 阈值 {threshold:.3f}，转 RAG")
            return None, True

        # 3.6 取原始问题 -> MySQL 查答案
        original_question = self.original_questions[best_idx]
        answer = self.mysql_client.fetch_answer(original_question)
        if not answer:
            self.logger.info(f"MySQL 未查到答案: {original_question}，转 RAG")
            return None, True

        # 3.7 回写缓存并返回
        self.redis_client.set_answer(query, answer)
        self.logger.info(f"BM25 命中: {query!r} ~= {original_question!r} (相似度 {score:.3f})")
        return answer, False


if __name__ == '__main__':
    bm25 = BM25Search(RedisClient(), MySQLClient())
    for q in ["大一新生去哪里报到？", "图书馆的开放时间", "怎么申请助学贷款", ""]:
        print(repr(q), "->", bm25.search(q))
