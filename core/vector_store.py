# xx_school_rag/core/vector_store.py
# 创建/加载 Milvus 向量库，灌入文档向量，执行混合检索 + 重排

# 导包
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值 -> 作为Milvus的主键.
import hashlib, sys, os
# 模型设备相关: 判断GPU是否可用， 为模型选择运行设备
import torch.cuda
# 导入数据加载和分割模块，用于准备数据
from utils.document_processor import process_documents
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库, 例如: 客户端, 数据类型, 检索请求, 排序器
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象, 即: 统一文档数据格式(含内容 + 元数据)
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断 -> 优化检索结果的相关性排序.
from sentence_transformers import CrossEncoder
# 导入项目配置类，用于获取项目配置信息
from base.config import Config
from base.logger import logger

conf = Config( )


def _bypass_proxy_for_milvus(host: str):
    """本机 Milvus 走本地回环，若系统设了 HTTP(S)_PROXY，需把 Milvus 主机加入 no_proxy，
    否则 gRPC 连接会被代理拦截（表现为 'server unavailable' / 502）。"""
    hosts = {host, "localhost", "127.0.0.1", "::1"}
    for var in ("no_proxy", "NO_PROXY", "no_grpc_proxy"):
        existing = {h.strip() for h in os.environ.get(var, "").split(",") if h.strip()}
        os.environ[var] = ",".join(sorted(existing | hosts))


_bypass_proxy_for_milvus(conf.MILVUS_HOST)

model_path = os.path.dirname(__file__)
project_root = os.path.dirname(model_path)
data_path = os.path.join(project_root, "data", "rag_data")


# todo 1. 定义VectorStore类: 封装向量库的核心操作 -> 集合管理, 文档入库, 混合检索, 结果处理.
class VectorStore:
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):

        # 1. 存储milvus数据库核心配置
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        # 2. 初始化日志实例
        self.logger = logger
        # 3. 选择模型运行设备（用字符串 'cuda'/'cpu'，新版 FlagEmbedding 不接受 torch.device 对象）
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'使用设备: {self.device}')

        # 4. 初始化BGE-M3模型 → 用于生成文档和查询的向量表示
        m3_path = os.path.join(project_root, 'models', 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(m3_path,  # # 模型本地路径
                                                         use_fp16=(self.device == 'cuda'), # # GPU时启用半精度计算(减少内存占用, 提升速度), CPU时禁用
                                                         device=self.device)
        # 5. 初始化BGE-Reranker重排序模型 → 优化检索结果相关性排序
        reranker_path = os.path.join(project_root, 'models', 'bge-reranker-large')
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        # 6. 获取BGE-M3稠密向量的维度: 固定输出1024维稠密向量.
        self.dense_dim = self.embedding_function.dim['dense']  # 稠密向量的维度: 1024

        # 7. 初始化Milvus客户端，建立milvus向量库连接
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 8. 调用'私有'方法 -> 创建新集合(若不存在) 或 加载已有集合(若存在)
        self._create_or_load_collection()

    #  todo 1.2 私有方法: 创建新集合(若不存在) 或 加载已有集合(若存在)
    def _create_or_load_collection(self):
        # 1. 检查集合是否存在
        if not self.client.has_collection(self.collection_name):
            # 不存在，则进行创建(字段+索引)
            # 2. 创建集合Schema，禁用自动ID(手动用文档哈希作为主键)，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 3. 添加字段
            # id字段
            schema.add_field(field_name='id', datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 父块字段
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 学科类别
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 时间戳
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 4. 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 4.1 为稠密向量字段添加 IVF_FLAT 索引，度量类型为(IP)内积
            index_params.add_index(field_name='dense_vector',
                                   index_name='dense_index',
                                   index_type="IVF_FLAT",
                                   metric_type="IP",
                                   params={"nlist": 128})
            # 4.2 为稀疏向量字段添加 HNSW 索引，度量类型为(IP)
            index_params.add_index(field_name='sparse_vector',
                                   index_name='sparse_index',
                                   index_type='SPARSE_INVERTED_INDEX',
                                   metric_type="IP",
                                   params={"drop_ratio_build": 0.2})  # 构建索引时,丢弃20%的低权重值(减少存储,噪声, 精度影响较小)

            # 5. 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            self.logger.info(f"集合 {self.collection_name} 创建成功")

        else:
            self.client.load_collection(self.collection_name)
            self.logger.info(f"集合 {self.collection_name} 加载成功")

    # todo 1.3 公有方法: 将文档(子块) 转换为 向量并插入Milvus
    def add_documents(self, documents):
        # 1. 提取子块文档内容列表
        texts = [doc.page_content for doc in documents]
        # 2. 使用BGE-M3模型对texts进行向量化
        embeddings = self.embedding_function(texts)
        # 3. 初始化空列表
        data = []
        # 4. 遍历每个文档，组装数据
        for i, doc in enumerate(documents):
            # 4.1 生成文档唯一的哈希ID
            # encode("uft-8"): 将字符串转成字节流 -> 因为MD5哈希需要字节.
            # hexdigest():      将哈希结果转为16进制字符串(32位, 适合作为ID)
            text_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

            # 4.2 处理稀疏向量: BGE-M3返回的稀疏向量是矩阵, 需要转换为Milvus支持的字典格式.
            # 4.2.1 初始化稀疏向量字典
            sparse_vector = {}
            # 4.2.2 获取第i个文档的稀疏向量行  → 获取矩阵的第i行
            row = embeddings["sparse"][[i]]
            # 4.2.3 获得稀疏向量的非零索引和对应的权重
            indics = row.indices
            weights = row.data
            # 组装稀疏向量字典
            for idx, value in zip(indics, weights):
                sparse_vector[idx] = value

            # 4.3 组装单条数据, 并加入到列表中 → 字段要和Schema完全对应
            data.append({
                "id": text_hash,  # 唯一ID(md5哈希)
                "text": doc.page_content,  # 文档内容
                "dense_vector": embeddings["dense"][i],  # 稠密向量(BGE-M3生成)
                "sparse_vector": sparse_vector,  # 稀疏向量(组装后的字典形式)
                "parent_id": doc.metadata["parent_id"],  # 父文档ID(从元数据获取, 例如: doc_0_parent_0)
                "parent_content": doc.metadata["parent_content"],  # 父文档内容(从元数据获取, 用于: 上下文补充)
                "source": doc.metadata.get("source", "unknown"),  # 学科类别(从元数据获取, 例如: ai)
                "timestamp": doc.metadata.get("timestamp", "unknown")  # 时间戳(从元数据获取)
            })

        # 5. 插入数据到Milvus -> 仅当data非空时执行.
        if data:
            # 使用upsert操作: 相同ID则更新, 不存在则插入(避免重复)
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 记录日志
            self.logger.info(f'已插入/更新 {len(data)} 条数据到集合 {self.collection_name}')

    # todo 1.4 公有方法: 混合检索(稠密 + 稀疏) + 重排序，返回精确父文档
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        # ---------------------- 1. 对输入查询进行预处理 --------------------------
        # 1. 使用BGE-M3模型对查询文本进行向量化
        query_embedding = self.embedding_function([query]) # [query]: embedding_function()方法期待一个列表作为输入
        # 2. 获取查询文本的稠密向量
        dense_query_vector = query_embedding["dense"][0]
        # 3. 获取查询文本的稀疏向量
        # 3.1 始化查询的稀疏向量字典
        sparse_query_vector = {}
        # 3.2  获取查询稀疏向量的第 0 行数据 -> 仅1个查询, 故取第0行.
        row = query_embedding['sparse'][[0]]
        # 3.3 获取稀疏向量的索引和权重
        indics = row.indices  # 索引列表
        weights = row.data  # 权重列表
        # 3.4 组装稀疏向量字典
        for idx, value in zip(indics, weights):
            sparse_query_vector[idx] = value

        # ---------------------- 2. 构建混合检索 --------------------------
        # 1. 构建索引过滤表达式，按学科过滤
        filter_expression = f"source == '{source_filter}'" if source_filter else ""
        # 2，构建稠密向量检索请求:  定义稠密向量的检索参数.
        dense_request = AnnSearchRequest(
            data=[dense_query_vector], # 查询向量, 列表格式.
            anns_field="dense_vector", # 检索的向量字段
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expression)
        # 3. 构建稀疏向量检索请求:  定义稀疏向量的检索参数.
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],  # 查询向量, 列表格式.
            anns_field="sparse_vector",  # 检索的向量字段
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expression)
        # 4. 创建加权排序器 -> 融合稠密和稀疏检索的结果, 按权重计算最终得分.
        ranker = WeightedRanker(1.0, 0.7)  # 稠密向量权重: 1.0, 稀疏向量权重: 0.7

        # ---------------------- 3. 执行混合检索 --------------------------
        results = self.client.hybrid_search( #  # 列表类型.
            collection_name=self.collection_name,  # 集合名
            reqs=[dense_request, sparse_request],  # 混合检索请求列表
            ranker=ranker,  # 加权排序器 -> 融合稠密和稀疏检索的结果, 按权重计算最终得分.
            limit=k,  # 返回Tok-K结果
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]  # 输出字段, 用于获取结果中的字段数据.
        )[0]  # 因为就1个查询, 所以返回第0个元素.

        # ---------------------- 4. 将查询结果转为list[document]对象 --------------------------
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]  # _doc_from_hit()方法返回Document对象.

        # ---------------------- 5. 去重父文档 --------------------------
        parent_docs = self._get_unique_parent_docs(sub_chunks)  # _get_unique_parent_docs()方法返回去重后的父文档列表.

        # ---------------------- 6. 进行重排序 --------------------------
        if len(parent_docs) < 2:
            # 只有0或1个文档是，无需重排序，直接返回
            ranked_parent_docs = parent_docs
        else:
            # 2个及以上文档时，执行重排序
            # 1. 构建'查询-文档'配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 2. 计算相关得分
            scores = self.reranker.predict(pairs)
            # 3. 按得分降序排序
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), key=lambda x: x[0], reverse=True)]

        # 返回重排序后的父文档列表 -> Top-M个父文档
        return ranked_parent_docs[:conf.CANDIDATE_M]

    # todo 1.5 私有方法，Milvus结果(hit) 转换为 LangChain Document对象.
    @staticmethod
    def _doc_from_hit(hit):
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )
    # todo 3.6 私有方法 -> 从子块列表中提取去重的父文档.
    @staticmethod
    def _get_unique_parent_docs(sub_chunks):
        # 1. 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 2. 初始化列表，用于存储唯一的父文档
        unique_docs = []
        # 3. 遍历所有子块
        for chunk in sub_chunks:
            # 3.1 获取当前子块的父块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 3.2 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 1. 创建新的Document对象，并添加到唯一父文档列表中
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                parent_contents.add(parent_content)
        # 4. 返回唯一父文档列表
        return unique_docs


if __name__ == '__main__':
    vector_store = VectorStore()
    # 灌库示例：vector_store.add_documents(process_documents(os.path.join(data_path, "报到_data")))
    query = '新生报到需要带什么材料'
    results = vector_store.hybrid_search_with_rerank(query, source_filter='报到')

    print(f'results: {results}')
    print(f'results数量: {len(results)}')