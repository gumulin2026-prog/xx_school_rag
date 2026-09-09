# xx_school_rag/utils/document_processor.py
# 该脚本用于: 递归加载 data/rag_data/<分类>_data/ 下的文档，做父子分层切分，供灌入 Milvus。

# 导包
import os
import sys
from datetime import datetime

# 把 utils/ 和项目根目录加入 sys.path，兼容「作为模块导入」和「直接运行本文件」两种方式
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
for _p in (_PROJECT_ROOT, _UTILS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 导入项目模块
from base.config import Config
from base.logger import logger
# 导入数据加载模块
from langchain_community.document_loaders import TextLoader
# 导入数据切分模块
from edu_text_spliter import ChineseRecursiveTextSplitter
from langchain_text_splitters import MarkdownTextSplitter

# 导入配置
conf = Config()

# 配置路径
model_path = os.path.dirname(__file__)
project_root = os.path.dirname(model_path)
data_path = os.path.join(project_root, "data", "rag_data")

# todo 1.建立文件扩展名到加载器类的映射
#   .md/.txt 用 TextLoader（避免 unstructured 依赖）；
#   pdf/docx/ppt/图片 加载器依赖较重（PyMuPDF/opencv/python-pptx/rapidocr），
#   缺失时对应格式自动跳过，不影响纯文本文档加载。
document_loaders = {
    ".txt": TextLoader,
    ".md":  TextLoader,
}
try:
    from .edu_document_loaders import OCRPDFLoader
    document_loaders[".pdf"] = OCRPDFLoader
except ImportError as e:
    logger.warning(f"PDF 加载器不可用（缺 PyMuPDF/opencv?）: {e}")
try:
    from .edu_document_loaders import OCRDOCLoader
    document_loaders[".docx"] = OCRDOCLoader
except ImportError as e:
    logger.warning(f"DOCX 加载器不可用（缺 python-docx?）: {e}")
try:
    from .edu_document_loaders import OCRPPTLoader
    document_loaders[".ppt"] = OCRPPTLoader
    document_loaders[".pptx"] = OCRPPTLoader
except ImportError as e:
    logger.warning(f"PPT 加载器不可用（缺 python-pptx?）: {e}")
try:
    from .edu_document_loaders import OCRIMGLoader
    document_loaders[".jpg"] = OCRIMGLoader
    document_loaders[".png"] = OCRIMGLoader
except ImportError as e:
    logger.warning(f"图片加载器不可用（缺 opencv/rapidocr?）: {e}")

# todo 2. 递归加载指定目录下的所有支持类型的文件，并 enrich（丰富）其元数据。
def load_documents_from_directory(directory_path=None):
    documents = []
    # 1. 获取支持的文件扩展名
    supported_extensions = document_loaders.keys()
    # 2. 从目录名提供"学科类别"原数据，
    source = os.path.basename(directory_path).replace('_data', "")
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # 3. 获取文件扩展名
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension in supported_extensions:
                try:
                    # 4. 根据文件扩展名获得对应的加载器，创建加载器实例
                    loader_class = document_loaders[file_extension]
                    if file_extension in (".txt", ".md"):
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    loaded_docs = loader.load()
                    # 5. 为加载的文档添加元数据
                    for doc in loaded_docs:
                        # 为文档添加学科类别元数据, 例如: ai
                        doc.metadata['source'] = source
                        # 为文档添加文件路径元数据
                        doc.metadata['file_path'] = file_path
                        # 为文档添加创建时间元数据
                        doc.metadata['created_at'] = datetime.now().isoformat()
                    documents.extend(loaded_docs)
                    logger.info(f"Loaded {file_name}")
                except Exception as e:
                    # 5.3.7 捕获异常 -> 输出异常信息, 继续处理下一个文件.
                    logger.error(f"Error loading {file_name}: {e}")
    return documents


# todo 3. 执行核心的分层切分策略
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    documents = load_documents_from_directory(directory_path)
    # 2. 初始化父块和子块的分割器
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 3. 创建用于存储子块的列表
    child_chunks = []

    # 4. 遍历加载好的list[document]列表
    for i, doc in enumerate(documents):
        # 4.1 判断是否为.md文件
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()
        is_markdown = (file_extension == '.md') # True or False
        # 4.2 提取文件类型，选择对应分割器
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        # 4.3 将list列表中的每一个document对象分割为父块 -> 大粒度, 保留较多上下文.
        parent_chunks = parent_splitter_to_use.split_documents([doc])
        # 5. 遍历每个父块，为父块增加元数据 → 父块切分成子块 → 小精度，用于精准匹配
        for j, parent_doc in enumerate(parent_chunks):
            # 5.1 为父块添加元数据
            parent_id = f"doc_{i}_parent_{j}"
            parent_doc.metadata["parent_id"] = parent_id
            # 5.2 将父块分割为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            # 5.3 遍历当前父块产生的子块，添加元数据并加入总列表
            for k, child_chunk in enumerate(sub_chunks):
                child_chunk.metadata["parent_id"] = parent_id # 记录父块id
                child_chunk.metadata["parent_content"] = parent_doc.page_content # 记录父块内容
                child_chunk.metadata["id"] = f"{parent_id}_child_{k}" # 记录子块id
                # 5.4 添加子块到总列表
                child_chunks.append(child_chunk)
    # 6. 记录日志
    logger.info(f"切分完成的子块总数: {len(child_chunks)}")
    return child_chunks

if __name__ == '__main__':
    # 进行切分（示例：报到分类）
    demo_dir = os.path.join(data_path, "报到_data")
    child_chunks = process_documents(demo_dir)
    print(f'切分完成的子块总数: {len(child_chunks)}')
    print(f'前1个子块详情: {child_chunks[0] if child_chunks else "未切分到任何子块"}')


