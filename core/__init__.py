# core/__init__.py
# xx_school_rag 核心模块。
# 采用惰性导入（PEP 562）：`import core` / `from core.bm25_search import ...` 本身很轻，
# 只有真正访问 core.RAGSystem / core.VectorStore 等重对象时，才加载 torch / milvus 等依赖。

import importlib  # 导入 importlib 模块，用于在运行时动态导入子模块

# 定义"类名 -> 子模块相对路径"的映射表，实现按需（惰性）加载
_LAZY = {
    "RAGSystem": ".rag_system",              # RAGSystem 类实际定义在 core/rag_system.py 中
    "MySQLClient": ".mysql_client",          # MySQLClient 类实际定义在 core/mysql_client.py 中
    "RedisClient": ".redis_client",          # RedisClient 类实际定义在 core/redis_client.py 中
    "BM25Search": ".bm25_search",            # BM25Search 类实际定义在 core/bm25_search.py 中
    "VectorStore": ".vector_store",          # VectorStore 类实际定义在 core/vector_store.py 中
    "QueryClassifier": ".query_classifier",  # QueryClassifier 类实际定义在 core/query_classifier.py 中
    "StrategySelector": ".strategy_selector",  # StrategySelector 类实际定义在 core/strategy_selector.py 中
    "LLMClient": ".llm_client",              # LLMClient 类实际定义在 core/llm_client.py 中
}

# __all__ 声明本包对外暴露的所有名称，取自 _LAZY 的所有键
__all__ = list(_LAZY)


# 模块级 __getattr__（PEP 562）：当访问 core.XXX 且 XXX 未在模块命名空间中时会被自动调用
def __getattr__(name):
    if name in _LAZY:  # 判断请求的属性名是否是我们支持的惰性导入项
        # 根据映射表动态导入对应的子模块（例如 ".rag_system" -> core.rag_system）
        module = importlib.import_module(_LAZY[name], __name__)
        # 从导入的子模块中取出同名的类/对象并返回
        return getattr(module, name)
    # 如果不是已知的惰性导入项，则按照 Python 规范抛出 AttributeError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
