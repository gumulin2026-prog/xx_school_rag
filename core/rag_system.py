# core/rag_system.py
# 该脚本用于: RAG 系统主流程整合，串联所有模块。
#
# 流程:
#   1. BM25 快速通道（Redis 精确缓存 / BM25 近似 / MySQL 取答案）命中即返回
#   2. 意图识别：通用知识 -> 直接问 LLM（不检索）；专业咨询 -> 进入 RAG 检索
#   3. 检索策略选择（直接 / HyDE / 子查询 / 回溯问题）改写查询后做混合检索 + 重排
#   4. 大模型基于检索到的上下文 + 对话历史生成最终答案

import os
import sys
import time

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from base.config import Config
from base.logger import logger
from core.mysql_client import MySQLClient
from core.redis_client import RedisClient
from core.bm25_search import BM25Search
from core.vector_store import VectorStore
from core.query_classifier import QueryClassifier
from core.strategy_selector import StrategySelector
from core.prompts import RAGPrompts
from core.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "你是 XX 大学迎新助手，负责解答新生关于报到、住宿、校园卡、食堂、选课、"
    "军训、快递、交通、图书馆、社团等方面的问题。请优先根据提供的资料准确回答；"
    "资料不足时如实说明，不要编造。"
)

_GENERAL_SYSTEM_PROMPT = "你是一个友好的助手，请根据你自己的知识简洁地回答问题。"

# 每个策略最多检索几路查询、最终合并保留几个文档块
_MAX_SUBQUERIES = 3
_MAX_MERGED_DOCS = 4


class RAGSystem:
    """RAG 系统主类，整合数据层、检索层、生成层。"""

    # todo 1. 初始化
    def __init__(self):
        self.logger = logger
        self.config = Config()
        # 数据层
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        # 意图识别（通用知识 / 专业咨询）
        self.query_classifier = QueryClassifier()
        # 检索层
        self.vector_store = VectorStore()
        self.strategy_selector = StrategySelector()
        # 生成层
        self.llm = LLMClient()
        self.prompts = RAGPrompts()
        self.logger.info("RAG 系统初始化完成")

    # ==================== 私有辅助方法 ====================
    @staticmethod
    def _format_history(history) -> str:
        """校验对话历史并格式化为字符串（只保留最近 5 轮）。"""
        if history is None or not isinstance(history, list):
            return ""
        history = history[-5:]
        for h in history:
            if not (isinstance(h, dict) and "question" in h and "answer" in h):
                logger.warning(f"无效的历史条目: {h}，忽略历史")
                return ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")
            return history_context
        return ""

    @staticmethod
    def _format_context(docs) -> str:
        """把检索到的文档拼成上下文。"""
        if not docs:
            logger.info("未检索到相关文档，上下文为空")
            return ""
        context = "\n\n".join([doc.page_content for doc in docs])
        logger.info(f"构建上下文完成，包含 {len(docs)} 个文档块")
        return context

    def _build_rag_prompt(self, context: str, history: str, question: str) -> str:
        return self.prompts.rag_prompt().format(
            context=context,
            history=history,
            question=question,
            phone=self.config.CUSTOMER_SERVICE_PHONE,
        )

    def _safe_llm_call(self, prompt: str, system_prompt: str, stream: bool):
        try:
            return self.llm.call(prompt=prompt, system_prompt=system_prompt,
                                 temperature=0.7, stream=stream)
        except Exception as e:
            logger.error(f"调用 LLM 失败: {e}")
            return (f"抱歉，处理你的问题时出错。"
                    f"请联系人工客服：{self.config.CUSTOMER_SERVICE_PHONE}")

    def _llm_rewrite(self, prompt: str, system_prompt: str):
        """给查询改写用：成功返回文本，失败返回 None（不返回兜底话术）。"""
        try:
            out = self.llm.call(prompt=prompt, system_prompt=system_prompt,
                                temperature=0.3, stream=False)
        except Exception as e:
            logger.error(f"查询改写 LLM 调用失败: {e}")
            return None
        out = (out or "").strip()
        if not out or out.startswith("抱歉"):
            return None
        return out

    # ==================== 检索策略 ====================
    def _rewrite_queries(self, query: str) -> tuple[list[str], str]:
        """按 strategy_selector 选出的策略把 query 改写成一路或多路检索文本。"""
        strategy = self.strategy_selector.select_strategy(query)
        self.logger.info(f"检索策略: {strategy}")

        if "假设" in strategy or "hyde" in strategy.lower():
            hypo = self._llm_rewrite(
                self.prompts.hyde_prompt().format(query=query),
                "你是知识助手，直接写出一段简短的假设性答案，不要说明。")
            return ([hypo] if hypo else [query]), strategy

        if "子查询" in strategy or "子问题" in strategy:
            raw = self._llm_rewrite(
                self.prompts.subquery_prompt().format(query=query),
                "你是查询分解助手，把复杂问题拆成几个简单子问题，每行一个，不要编号。")
            if raw:
                subs = [line.strip().lstrip("0123456789.、-　 ").strip()
                        for line in raw.splitlines() if line.strip()]
                subs = [s for s in subs if s][:_MAX_SUBQUERIES]
                if subs:
                    return subs, strategy
            return [query], strategy

        if "回溯" in strategy or "退一步" in strategy:
            simpler = self._llm_rewrite(
                self.prompts.backtracking_prompt().format(query=query),
                "你是问题简化助手，只输出简化后的一个问题。")
            return ([simpler] if simpler else [query]), strategy

        # 默认：直接检索
        return [query], strategy

    def _retrieve(self, query: str, source_filter: str = None):
        """按策略改写查询 -> 逐路混合检索 -> 合并去重。返回 (docs, strategy)。"""
        search_texts, strategy = self._rewrite_queries(query)
        self.logger.info(f"检索文本({len(search_texts)}路): {search_texts}")

        seen, merged = set(), []
        for text in search_texts:
            for doc in self.vector_store.hybrid_search_with_rerank(
                    text, k=self.config.RETRIEVAL_K, source_filter=source_filter):
                key = doc.page_content
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged[:_MAX_MERGED_DOCS], strategy

    # ==================== 核心方法 ====================
    def generate_answer(self, query: str, source_filter: str = None,
                        history: list = None, stream: bool = False):
        """
        生成答案主入口。
        :param query: 用户问题
        :param source_filter: 分类过滤（对应 config.VALID_SOURCES）
        :param history: 对话历史 [{"question":..., "answer":...}]
        :param stream: 是否流式输出
        :return: 答案字符串，或流式模式下的生成器
        """
        start_time = time.time()
        self.logger.info(f"处理查询: '{query}', 过滤: {source_filter}")

        history_str = self._format_history(history)

        # 1. BM25 快速通道
        answer, _need_rag = self.bm25_search.search(query)
        if answer:
            self.logger.info(f"BM25 命中，耗时 {time.time() - start_time:.2f}s")
            if stream:
                def answer_gen():
                    yield answer
                return answer_gen()
            return answer

        # 2. 意图识别：通用知识 -> 直接问 LLM（不检索）
        category = self.query_classifier.predict_category(query)
        self.logger.info(f"意图识别: {category}")
        if category == "通用知识":
            prompt = self._build_rag_prompt(context="", history=history_str, question=query)
            answer = self._safe_llm_call(prompt, _GENERAL_SYSTEM_PROMPT, stream)
            self.logger.info(f"通用知识直答，耗时 {time.time() - start_time:.2f}s")
            return answer

        # 3. 专业咨询 -> 策略改写 + RAG 检索
        self.logger.info("专业咨询，执行 RAG 检索")
        docs, _strategy = self._retrieve(query, source_filter)
        context = self._format_context(docs)

        prompt = self._build_rag_prompt(context=context, history=history_str, question=query)
        answer = self._safe_llm_call(prompt, _SYSTEM_PROMPT, stream)
        self.logger.info(f"RAG 处理完成，耗时 {time.time() - start_time:.2f}s")
        return answer


if __name__ == '__main__':
    rag_system = RAGSystem()
    for q, sf in [("军训一般安排在什么时候？", "军训"),
                  ("报到当天的流程、需要带的材料、以及宿舍怎么分配", "报到")]:
        print("\nQ:", q)
        print("A:", rag_system.generate_answer(q, source_filter=sf))
