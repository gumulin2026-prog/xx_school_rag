# core/__init__.py
# xx_school_rag 核心模块。
# 采用惰性导入（PEP 562）：`import core` / `from core.bm25_search import ...` 本身很轻，
# 只有真正访问 core.RAGSystem / core.VectorStore 等重对象时，才加载 torch / milvus 等依赖。

import importlib

_LAZY = {
    "RAGSystem": ".rag_system",
    "MySQLClient": ".mysql_client",
    "RedisClient": ".redis_client",
    "BM25Search": ".bm25_search",
    "VectorStore": ".vector_store",
    "QueryClassifier": ".query_classifier",
    "StrategySelector": ".strategy_selector",
    "LLMClient": ".llm_client",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
