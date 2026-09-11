# core/rag_system.py
# 该脚本用于: RAG 系统主流程整合，串联所有模块。
#
# 流程:
#   1. BM25 快速通道（Redis 精确缓存 / BM25 近似 / MySQL 取答案）命中即返回
#   2. 意图识别：通用知识 -> 直接问 LLM（不检索）；专业咨询 -> 进入 RAG 检索
#   3. 检索策略选择（直接 / HyDE / 子查询 / 回溯问题）改写查询后做混合检索 + 重排
#   4. 大模型基于检索到的上下文 + 对话历史生成最终答案

import os     # 路径拼接
import sys    # 修改模块搜索路径
import time   # 计时，统计处理耗时

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件（rag_system.py）所在目录
project_root = os.path.dirname(current_dir)  # 项目根目录
sys.path.insert(0, project_root)  # 将项目根目录加入模块搜索路径，便于以脚本方式直接运行

from base.config import Config  # 项目统一配置类
from base.logger import logger  # 项目统一日志对象
from core.mysql_client import MySQLClient  # MySQL 数据访问层
from core.redis_client import RedisClient  # Redis 缓存访问层
from core.bm25_search import BM25Search  # BM25 快速通道搜索
from core.vector_store import VectorStore  # 向量库检索（混合检索 + 重排）
from core.query_classifier import QueryClassifier  # 意图分类器（通用知识 / 专业咨询）
from core.strategy_selector import StrategySelector  # 检索策略选择器
from core.prompts import RAGPrompts  # 各类 Prompt 模板集合
from core.llm_client import LLMClient  # 大模型调用客户端

_SYSTEM_PROMPT = (  # RAG 检索场景下使用的系统角色设定
    "你是 XX 大学迎新助手，负责解答新生关于报到、住宿、校园卡、食堂、选课、"
    "军训、快递、交通、图书馆、社团等方面的问题。请优先根据提供的资料准确回答；"
    "资料不足时如实说明，不要编造。"
)

_GENERAL_SYSTEM_PROMPT = "你是一个友好的助手，请根据你自己的知识简洁地回答问题。"  # 通用知识场景（不检索）下使用的系统角色设定

# 每个策略最多检索几路查询、最终合并保留几个文档块
_MAX_SUBQUERIES = 3     # 子查询策略下最多拆分出的子问题数量
_MAX_MERGED_DOCS = 4    # 多路检索结果合并去重后最多保留的文档块数量


class RAGSystem:
    """RAG 系统主类，整合数据层、检索层、生成层。"""

    # todo 1. 初始化
    def __init__(self):
        self.logger = logger  # 保存日志对象为实例属性
        self.config = Config()  # 实例化配置对象
        # 数据层
        self.mysql_client = MySQLClient()  # MySQL 客户端，负责知识库和对话历史的持久化
        self.redis_client = RedisClient()  # Redis 客户端，负责缓存
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)  # 快速通道搜索器，依赖 Redis 和 MySQL
        # 意图识别（通用知识 / 专业咨询）
        self.query_classifier = QueryClassifier()  # 加载 BERT 意图分类模型
        # 检索层
        self.vector_store = VectorStore()  # 向量库客户端，负责混合检索 + 重排
        self.strategy_selector = StrategySelector()  # 检索策略选择器
        # 生成层
        self.llm = LLMClient()  # 大模型调用客户端
        self.prompts = RAGPrompts()  # Prompt 模板集合
        self.logger.info("RAG 系统初始化完成")  # 记录初始化完成日志

    # ==================== 私有辅助方法 ====================
    @staticmethod
    def _format_history(history) -> str:
        """校验对话历史并格式化为字符串（只保留最近 5 轮）。"""
        if history is None or not isinstance(history, list):  # 历史为空或类型不对，直接视为无历史
            return ""
        history = history[-5:]  # 只保留最近 5 轮，避免上下文过长
        for h in history:  # 逐条校验历史记录格式
            if not (isinstance(h, dict) and "question" in h and "answer" in h):  # 必须是包含 question/answer 键的字典
                logger.warning(f"无效的历史条目: {h}，忽略历史")  # 格式不合法则记录警告
                return ""  # 整体历史作废，返回空字符串（保守处理）
        if history:  # 校验通过且非空
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]  # 将每轮问答拼成 "Q: ...\nA: ..." 格式
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")  # 记录日志（截断前100字符）
            return history_context  # 返回拼接后的历史文本
        return ""  # 空历史返回空字符串

    @staticmethod
    def _format_context(docs) -> str:
        """把检索到的文档拼成上下文。"""
        if not docs:  # 未检索到任何文档
            logger.info("未检索到相关文档，上下文为空")  # 记录日志
            return ""  # 返回空字符串
        context = "\n\n".join([doc.page_content for doc in docs])  # 用两个换行符拼接各文档块内容
        logger.info(f"构建上下文完成，包含 {len(docs)} 个文档块")  # 记录拼接的文档块数量
        return context  # 返回拼接后的上下文文本

    def _build_rag_prompt(self, context: str, history: str, question: str) -> str:
        return self.prompts.rag_prompt().format(  # 用 RAG 场景的 Prompt 模板填充占位符
            context=context,    # 检索到的上下文资料
            history=history,    # 格式化后的对话历史
            question=question,  # 用户当前问题
            phone=self.config.CUSTOMER_SERVICE_PHONE,  # 人工客服电话，供无法回答时兜底展示
        )

    def _safe_llm_call(self, prompt: str, system_prompt: str, stream: bool):
        try:
            return self.llm.call(prompt=prompt, system_prompt=system_prompt,
                                 temperature=0.7, stream=stream)  # 调用大模型生成最终答案（温度稍高，回答更自然）
        except Exception as e:  # 捕获调用过程中可能出现的任何异常
            logger.error(f"调用 LLM 失败: {e}")  # 记录错误日志
            return (f"抱歉，处理你的问题时出错。"
                    f"请联系人工客服：{self.config.CUSTOMER_SERVICE_PHONE}")  # 返回友好的兜底提示

    def _llm_rewrite(self, prompt: str, system_prompt: str):
        """给查询改写用：成功返回文本，失败返回 None（不返回兜底话术）。"""
        try:
            out = self.llm.call(prompt=prompt, system_prompt=system_prompt,
                                temperature=0.3, stream=False)  # 低温度调用大模型做查询改写，结果更稳定
        except Exception as e:  # 捕获调用过程中可能出现的任何异常
            logger.error(f"查询改写 LLM 调用失败: {e}")  # 记录错误日志
            return None  # 改写失败时返回 None，交由调用方回退到原始查询
        out = (out or "").strip()  # 去除首尾空白，None 时兜底为空字符串
        if not out or out.startswith("抱歉"):  # 结果为空 或 命中兜底话术，说明改写无效
            return None  # 视为改写失败
        return out  # 返回改写后的文本

    # ==================== 检索策略 ====================
    def _rewrite_queries(self, query: str) -> tuple[list[str], str]:
        """按 strategy_selector 选出的策略把 query 改写成一路或多路检索文本。"""
        strategy = self.strategy_selector.select_strategy(query)  # 调用大模型判断应使用哪种检索增强策略
        self.logger.info(f"检索策略: {strategy}")  # 记录选中的策略名称

        if "假设" in strategy or "hyde" in strategy.lower():  # 命中 HyDE（假设性答案）策略
            hypo = self._llm_rewrite(
                self.prompts.hyde_prompt().format(query=query),  # 用 HyDE 模板填充查询
                "你是知识助手，直接写出一段简短的假设性答案，不要说明。")  # 系统提示：只输出假设答案
            return ([hypo] if hypo else [query]), strategy  # 改写成功则用假设答案检索，失败则回退用原查询

        if "子查询" in strategy or "子问题" in strategy:  # 命中子查询拆分策略
            raw = self._llm_rewrite(
                self.prompts.subquery_prompt().format(query=query),  # 用子查询模板填充查询
                "你是查询分解助手，把复杂问题拆成几个简单子问题，每行一个，不要编号。")  # 系统提示：逐行输出子问题
            if raw:  # 改写成功，拿到大模型返回的多行文本
                subs = [line.strip().lstrip("0123456789.、-　 ").strip()
                        for line in raw.splitlines() if line.strip()]  # 按行拆分，去掉可能的编号前缀（数字/点/顿号/短横线/空格）
                subs = [s for s in subs if s][:_MAX_SUBQUERIES]  # 过滤空行，并限制最多子查询数量
                if subs:  # 拆分出了有效子查询
                    return subs, strategy  # 用这些子查询分别检索
            return [query], strategy  # 改写失败或未拆出子查询，回退用原查询

        if "回溯" in strategy or "退一步" in strategy:  # 命中回溯问题（Step-back）策略
            simpler = self._llm_rewrite(
                self.prompts.backtracking_prompt().format(query=query),  # 用回溯模板填充查询
                "你是问题简化助手，只输出简化后的一个问题。")  # 系统提示：只输出简化后的问题
            return ([simpler] if simpler else [query]), strategy  # 改写成功则用简化问题检索，失败则回退用原查询

        # 默认：直接检索
        return [query], strategy  # 未命中以上任何策略，直接用原查询检索

    def _retrieve(self, query: str, source_filter: str = None):
        """按策略改写查询 -> 逐路混合检索 -> 合并去重。返回 (docs, strategy)。"""
        search_texts, strategy = self._rewrite_queries(query)  # 根据策略得到一路或多路检索文本
        self.logger.info(f"检索文本({len(search_texts)}路): {search_texts}")  # 记录本次实际检索用的文本

        seen, merged = set(), []  # seen: 已出现过的文档内容集合（去重用）；merged: 合并后的文档列表
        for text in search_texts:  # 遍历每一路检索文本
            for doc in self.vector_store.hybrid_search_with_rerank(
                    text, k=self.config.RETRIEVAL_K, source_filter=source_filter):  # 对每路文本执行混合检索 + 重排
                key = doc.page_content  # 用文档内容本身作为去重的键
                if key not in seen:  # 尚未出现过的文档才保留
                    seen.add(key)  # 标记为已出现
                    merged.append(doc)  # 加入合并结果
        return merged[:_MAX_MERGED_DOCS], strategy  # 只保留前 N 个文档块，连同策略名一起返回

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
        start_time = time.time()  # 记录处理开始时间，用于统计耗时
        self.logger.info(f"处理查询: '{query}', 过滤: {source_filter}")  # 记录本次查询及分类过滤条件

        history_str = self._format_history(history)  # 校验并格式化对话历史为字符串

        # 1. BM25 快速通道
        answer, _need_rag = self.bm25_search.search(query)  # 先尝试缓存/BM25 快速匹配（_need_rag 未再使用，answer 判空即可知晓是否命中）
        if answer:  # 快速通道命中
            self.logger.info(f"BM25 命中，耗时 {time.time() - start_time:.2f}s")  # 记录命中及耗时
            if stream:  # 若调用方要求流式输出
                def answer_gen():  # 定义一个简单的生成器，把已有的完整答案包装成"单块流式输出"
                    yield answer
                return answer_gen()  # 返回生成器，保持接口一致性
            return answer  # 非流式模式直接返回完整答案

        # 2. 意图识别：通用知识 -> 直接问 LLM（不检索）
        category = self.query_classifier.predict_category(query)  # 用 BERT 分类器判断问题类型
        self.logger.info(f"意图识别: {category}")  # 记录识别结果
        if category == "通用知识":  # 通用知识类问题无需检索，直接让大模型凭自身知识回答
            prompt = self._build_rag_prompt(context="", history=history_str, question=query)  # 上下文置空，仍复用同一套模板
            answer = self._safe_llm_call(prompt, _GENERAL_SYSTEM_PROMPT, stream)  # 用通用系统提示调用大模型
            self.logger.info(f"通用知识直答，耗时 {time.time() - start_time:.2f}s")  # 记录耗时
            return answer  # 返回大模型生成的答案（或流式生成器）

        # 3. 专业咨询 -> 策略改写 + RAG 检索
        self.logger.info("专业咨询，执行 RAG 检索")  # 记录进入 RAG 检索分支
        docs, _strategy = self._retrieve(query, source_filter)  # 按策略改写查询并执行混合检索，得到候选文档（_strategy 仅用于日志已在内部记录）
        context = self._format_context(docs)  # 把检索到的文档拼接成上下文文本

        prompt = self._build_rag_prompt(context=context, history=history_str, question=query)  # 构建包含检索上下文的完整 Prompt
        answer = self._safe_llm_call(prompt, _SYSTEM_PROMPT, stream)  # 用迎新助手系统提示调用大模型生成答案
        self.logger.info(f"RAG 处理完成，耗时 {time.time() - start_time:.2f}s")  # 记录整体处理耗时
        return answer  # 返回最终答案（或流式生成器）


if __name__ == '__main__':  # 直接运行本文件时执行的自测代码
    rag_system = RAGSystem()  # 实例化 RAG 系统（会依次初始化各层客户端和模型）
    for q, sf in [("军训一般安排在什么时候？", "军训"),
                  ("报到当天的流程、需要带的材料、以及宿舍怎么分配", "报到")]:  # 遍历若干测试问题及其分类过滤条件
        print("\nQ:", q)  # 打印问题
        print("A:", rag_system.generate_answer(q, source_filter=sf))  # 打印生成的答案
