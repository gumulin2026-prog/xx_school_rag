# xx_shcool_rag/base/config.py
# 该脚本用于: 配置管理。
#   - 以项目根目录的 config.ini 为准，环境变量 / .env 可覆盖对应项。
#   - fallback 仅在 config.ini 缺失该项时生效。

import ast
import configparser
import os
import warnings

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件（存放 API Key 等敏感信息）
load_dotenv()


# todo 1. 配置文件路径计算
def get_project_root():
    """获取项目根目录的绝对路径（.../xx_shcool_rag），基于当前文件位置推算。"""
    current_file = os.path.abspath(__file__)        # .../xx_shcool_rag/base/config.py
    current_dir = os.path.dirname(current_file)     # .../xx_shcool_rag/base
    project_root = os.path.dirname(current_dir)     # .../xx_shcool_rag
    return project_root


PROJECT_ROOT = get_project_root()
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'config.ini')


# todo 2. 配置解析类
class Config:
    def __init__(self, config_file=CONFIG_FILE_PATH):
        # 创建配置解析器（支持 ${section:key} 插值）
        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        # 读取配置文件（文件不存在也不报错，全部走 fallback）
        self.config.read(config_file, encoding='utf-8')

        # ----- 目录路径 -----
        self.PROJECT_ROOT = PROJECT_ROOT
        self.DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
        self.MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
        self.LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
        # 文档加载器所在目录（utils/edu_document_loaders/__init__.py 会加入 sys.path）
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(PROJECT_ROOT, 'utils')

        # ----- 日志（路径取自 config.ini [log] log_file，相对项目根目录） -----
        log_file = self.config.get('log', 'log_file', fallback='logs/app.log')
        self.LOG_FILE = os.path.normpath(os.path.join(PROJECT_ROOT, log_file))
        self.LOG_DIR = os.path.dirname(self.LOG_FILE)

        # ----- MySQL 配置 -----
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        self.MYSQL_PORT = int(os.getenv('MYSQL_PORT', self.config.get('mysql', 'port', fallback='3306')))
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='xx_school_rag'))

        # ----- Redis 配置 -----
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback='6379')))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback='1')))

        # ----- Milvus 配置 -----
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        self.MILVUS_DATABASE_NAME = os.getenv(
            'MILVUS_DATABASE_NAME',
            self.config.get('milvus', 'database_name', fallback='xx_school_rag'))
        self.MILVUS_COLLECTION_NAME = os.getenv(
            'MILVUS_COLLECTION_NAME',
            self.config.get('milvus', 'collection_name', fallback='xx_school_rag_collection'))

        # ----- LLM 配置（DeepSeek，兼容 OpenAI 接口） -----
        # 密钥/base_url 优先从 .env 读取，其次 config.ini
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='deepseek-chat')
        self.LLM_BASE_URL = (os.getenv('DEEPSEEK_BASE_URL')
                             or self.config.get('llm', 'base_url', fallback='https://api.deepseek.com'))
        self.LLM_API_KEY = os.getenv('DEEPSEEK_API_KEY') or self.config.get('llm', 'api_key', fallback='')
        if not self.LLM_API_KEY:
            # 当前阶段（BM25 / MySQL / Redis 缓存层）不需要 LLM，缺 key 仅告警不中断
            warnings.warn(
                "未检测到 DEEPSEEK_API_KEY（.env 或 config.ini [llm] api_key），"
                "接入 RAG 生成前需补上。",
                RuntimeWarning,
                stacklevel=2,
            )
            self.LLM_API_KEY = None

        # ----- 检索参数 -----
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=3)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # ----- 应用配置 -----
        raw_sources = self.config.get('app', 'valid_sources', fallback='[]')
        try:
            self.VALID_SOURCES = ast.literal_eval(raw_sources)
        except (ValueError, SyntaxError):
            warnings.warn(f"config.ini [app] valid_sources 解析失败: {raw_sources!r}", RuntimeWarning)
            self.VALID_SOURCES = []
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')

        # ----- BM25 配置 -----
        self.THRESHOLD = self.config.getfloat('bm25', 'threshold', fallback=0.85)


# 创建全局单例配置对象（供其他模块导入使用）
config = Config()


# todo 3. 测试代码
if __name__ == '__main__':
    print(f"项目根目录: {config.PROJECT_ROOT}")
    print(f"数据目录:   {config.DATA_DIR}")
    print(f"MySQL: {config.MYSQL_HOST}:{config.MYSQL_PORT}, db={config.MYSQL_DATABASE}")
    print(f"Redis: {config.REDIS_HOST}:{config.REDIS_PORT}, db={config.REDIS_DB}")
    print(f"Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT}, "
          f"db={config.MILVUS_DATABASE_NAME}, coll={config.MILVUS_COLLECTION_NAME}")
    print(f"LLM: model={config.LLM_MODEL}, base_url={config.LLM_BASE_URL}")
    print(f"LLM API Key: {config.LLM_API_KEY[:8] + '...' if config.LLM_API_KEY else '未设置'}")
    print(f"BM25 阈值: {config.THRESHOLD}")
    print(f"有效分类: {config.VALID_SOURCES}")
    print(f"日志文件: {config.LOG_FILE}")
