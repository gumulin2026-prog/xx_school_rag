# xx_school_rag/scripts/init_milvus.py
# 该脚本用于: 初始化 Milvus —— 建库、建集合、把 data/rag_data/<分类>_data/ 下的文档
#             切分、向量化并灌入向量库。可重复执行（按内容哈希 upsert，不会重复）。
#
# 运行: python scripts/init_milvus.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.config import config
from base.logger import logger


def ensure_database():
    """Milvus 不会自动建 database，这里先确保目标 database 存在。"""
    from pymilvus import MilvusClient
    from core.vector_store import _bypass_proxy_for_milvus

    _bypass_proxy_for_milvus(config.MILVUS_HOST)
    admin = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")
    dbs = admin.list_databases()
    if config.MILVUS_DATABASE_NAME not in dbs:
        admin.create_database(config.MILVUS_DATABASE_NAME)
        logger.info(f"已创建 Milvus database: {config.MILVUS_DATABASE_NAME}")
    else:
        logger.info(f"Milvus database 已存在: {config.MILVUS_DATABASE_NAME}")
    admin.close()


def main():
    ensure_database()

    from core.vector_store import VectorStore
    from utils.document_processor import process_documents

    store = VectorStore()

    rag_root = os.path.join(config.DATA_DIR, "rag_data")
    total = 0
    for name in sorted(os.listdir(rag_root)):
        sub = os.path.join(rag_root, name)
        if not os.path.isdir(sub):
            continue
        chunks = process_documents(sub)
        if not chunks:
            logger.warning(f"{name}: 未切分到子块，跳过")
            continue
        store.add_documents(chunks)
        total += len(chunks)
        logger.info(f"{name}: 已灌入 {len(chunks)} 个子块")

    logger.info(f"Milvus 初始化完成，共灌入 {total} 个子块到集合 {config.MILVUS_COLLECTION_NAME}")

    # 冒烟检索
    for q, sf in [("新生报到需要带什么材料", "报到"), ("宿舍能用电热水壶吗", "住宿")]:
        docs = store.hybrid_search_with_rerank(q, source_filter=sf)
        logger.info(f"[检索] {q} -> {len(docs)} 条；首条: "
                    f"{docs[0].page_content[:60] if docs else '无'}")


if __name__ == "__main__":
    main()
