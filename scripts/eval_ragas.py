# xx_school_rag/scripts/eval_ragas.py
# 该脚本用于: 用 RAGAS 评估 RAG 系统「检索 + 生成」链路的效果。
#
#   - 测试集: 直接取 qa_knowledge 表里现成的 (question, answer, category)，
#     answer 作为参考答案(reference)。
#   - 刻意绕开 BM25 精确/近似缓存与意图分类，只跑 core.rag_system 里的
#     "检索 -> 拼 prompt -> 调用大模型生成" 这两步。因为测试集问题本来就和
#     BM25 问题库里的问题逐字相同，如果走完整 generate_answer()，BM25 会
#     100% 命中缓存直接返回库里的原答案，根本测不出 RAG 检索+生成的真实水平。
#   - 指标: faithfulness / answer_relevancy / context_precision / context_recall
#     裁判 LLM 复用项目自己的 DeepSeek 配置，embedding 复用项目里已经在用的 BGE-M3
#     （不引入额外的外部依赖/额外的 API Key）。
#
#   注意: data/rag_data/ 目前只有 报到/住宿/军训/选课/食堂/校园生活 六个类别的
#   真实资料文档，qa_knowledge 里另外 6 个分类(校园卡/快递/交通/学习/社团/图书馆)
#   在向量库里找不到对应资料，这些分类的分数偏低是预期内的真实发现，不是 bug。
#
# 运行: python scripts/eval_ragas.py [样本数，默认全部30条]
# 输出: data/eval/ragas_report.csv（逐条明细） + data/eval/ragas_report.json（汇总）

import sys
import types


def _patch_missing_vertexai_shim():
    """在 import ragas 之前，往 sys.modules 里塞一个假的
    langchain_community.chat_models.vertexai 模块。

    背景: 已安装的 ragas==0.4.3 在 import 时会无条件执行
    `from langchain_community.chat_models.vertexai import ChatVertexAI`，
    但本项目锁定的 langchain-community==0.4.2 已经把这个子模块整个移除了
    （该包已进入 sunset 状态，各家模型的集成陆续被拆到独立包），导致
    `import ragas` 直接抛 ModuleNotFoundError。我们用不到 VertexAI，
    这里垫一个占位模块把这次 import 糊弄过去，不需要改动已安装的第三方包。
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    fake_module = types.ModuleType(module_name)

    class ChatVertexAI:  # 占位类，永远不会被真正实例化/调用
        pass

    fake_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = fake_module


_patch_missing_vertexai_shim()

import json
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

# 路径配置：把项目根目录加入 sys.path，便于以脚本方式直接运行本文件
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from base.config import config
from base.logger import logger
from core.mysql_client import MySQLClient
from core.rag_system import RAGSystem, _SYSTEM_PROMPT

_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

OUT_DIR = os.path.join(_ROOT, "data", "eval")
OUT_CSV = os.path.join(OUT_DIR, "ragas_report.csv")
OUT_JSON = os.path.join(OUT_DIR, "ragas_report.json")


class _SingleCompletionChatOpenAI(ChatOpenAI):
    """DeepSeek 的 Chat Completions 接口只支持 n=1（一次请求只返回一个结果），
    但 RAGAS 的部分指标（例如 answer_relevancy 默认会让裁判 LLM 一次生成
    3 个候选问题取平均，context_precision/context_recall 内部也会做类似的
    多次采样）会显式请求 n>1，导致 DeepSeek 直接返回 400 Invalid n value，
    对应那条样本的这项指标就变成 NaN。

    这里在请求发出前把 n 强制锁定为 1：牺牲掉"多次采样取多数"带来的稳健性，
    换取所有样本、所有指标都能跑出完整分数——RAGAS 自身对"返回的生成数量少于
    请求数量"这种情况有优雅降级（日志会打印 "LLM returned 1 generations
    instead of requested 3. Proceeding with 1 generations."），并不会报错。"""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _BGEEmbeddingsAdapter:
    """把项目里的 BGEM3EmbeddingFunction 包装成 LangChain Embeddings 接口
    （embed_query / embed_documents），供 ragas 的 answer_relevancy 指标
    计算"生成的问题 vs 原始问题"的相似度时使用。"""

    def __init__(self, embedding_function):
        self._fn = embedding_function

    def embed_documents(self, texts):
        vectors = self._fn(list(texts))["dense"]  # BGE-M3 返回 {"dense": [...], "sparse": ...}
        return [[float(x) for x in vec] for vec in vectors]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def _load_testset(limit=None):
    """从 qa_knowledge 表读取测试集：[(question, answer, category), ...]。"""
    client = MySQLClient()
    client.cursor.execute("SELECT question, answer, category FROM qa_knowledge ORDER BY id")
    rows = client.cursor.fetchall()
    client.close()
    return rows[:limit] if limit else rows


def _run_rag_pipeline(rag_system: RAGSystem, question: str):
    """只跑"检索 + 生成"两步，绕开 BM25 缓存和意图分类（原因见文件头注释）。

    :return: (retrieved_contexts, answer) —— retrieved_contexts 是检索到的
             各文档块原文列表；answer 是大模型基于这些文档块生成的回答。
    """
    docs, _strategy = rag_system._retrieve(question, source_filter=None)
    retrieved_contexts = [doc.page_content for doc in docs]
    context_text = rag_system._format_context(docs)
    prompt = rag_system._build_rag_prompt(context=context_text, history="", question=question)
    answer = rag_system._safe_llm_call(prompt, _SYSTEM_PROMPT, stream=False)
    return retrieved_contexts, answer


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    logger.info("加载测试集（qa_knowledge 表）...")
    rows = _load_testset(limit)
    logger.info(f"测试集加载完成，共 {len(rows)} 条")

    logger.info("初始化 RAG 系统（会加载 BERT / BGE-M3 / Reranker，稍等）...")
    rag_system = RAGSystem()

    logger.info("逐条跑「检索 + 生成」，收集 RAGAS 需要的样本...")
    samples = []
    categories = []
    for i, (question, reference, category) in enumerate(rows, 1):
        logger.info(f"[{i}/{len(rows)}] ({category}) {question}")
        retrieved_contexts, answer = _run_rag_pipeline(rag_system, question)
        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=retrieved_contexts or ["（未检索到任何相关文档）"],  # ragas 要求非空列表
            response=answer or "",
            reference=reference,
        ))
        categories.append(category)

    dataset = EvaluationDataset(samples=samples)

    logger.info("初始化 RAGAS 裁判 LLM（复用项目的 DeepSeek 配置）与 embedding（复用 BGE-M3）...")
    judge_llm = LangchainLLMWrapper(_SingleCompletionChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0,
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(
        _BGEEmbeddingsAdapter(rag_system.vector_store.embedding_function)
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    logger.info("开始用 RAGAS 打分（要多次调用 LLM 当裁判，需要几分钟，请耐心等待）...")
    result = evaluate(dataset, metrics=metrics)

    df = result.to_pandas()
    df["category"] = categories

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    overall = {name: float(df[name].mean()) for name in _METRIC_NAMES}
    by_category = {
        cat: {name: float(sub[name].mean()) for name in _METRIC_NAMES}
        for cat, sub in df.groupby("category")
    }
    report = {"n_samples": len(df), "overall": overall, "by_category": by_category}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== 整体平均分 =====")
    for name, score in overall.items():
        print(f"{name}: {score:.3f}")
    print("\n===== 按分类平均分（数值越低说明该分类知识库资料/检索效果越差） =====")
    for cat, scores in by_category.items():
        print(cat, {name: round(score, 3) for name, score in scores.items()})
    print(f"\n逐条明细已保存: {OUT_CSV}")
    print(f"汇总报告已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
