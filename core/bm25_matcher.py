# xx_school_rag/core/bm25_matcher.py
# 该脚本用于: 中文分词 + BM25 近似匹配。
#   - tokenize(): jieba 精确分词（加载领域词典 data/user_dict.txt），去停用词 / 空白
#   - best_match(): 用 Redis 全量分词问题库现建 BM25（不持久化），
#                   返回 (最佳下标, 归一化相似度)
#
# 归一化方式(按需求 3A): normalized = 语料中最高 BM25 分 / query 对自身的 BM25 分。
#   实现: 把 query 自身作为一篇临时文档追加到语料再建 BM25，
#         追加文档的得分即“query 对自身打分”，用它做分母；结果截断到 [0, 1]。
#   （不能用「单文档语料」算自打分——那样每个词 df=N，IDF 变负，分数恒 <= 0。）

import logging  # 标准日志模块，用于设置 jieba 的日志级别
from pathlib import Path  # 面向对象的文件路径处理工具

import jieba  # 中文分词库
from rank_bm25 import BM25Okapi  # BM25 算法实现，用于计算文本相似度打分

from base.config import config  # 项目全局配置单例（含数据目录、阈值等）
from base.logger import logger  # 项目统一日志对象

jieba.setLogLevel(logging.WARNING)  # 静默 "Building prefix dict..." 等启动噪声。不影响功能，只是让终端输出更干净。

_DATA_DIR = Path(config.DATA_DIR)  # 数据文件根目录（Path 对象，来自配置）
_STOPWORDS_FILE = _DATA_DIR / "stopwords.txt"  # 停用词文件路径
_USER_DICT_FILE = _DATA_DIR / "user_dict.txt"  # jieba 领域自定义词典文件路径


def _load_stopwords() -> set[str]:  # 从文件加载停用词集合，返回 set 便于快速查找
    try:
        text = _STOPWORDS_FILE.read_text(encoding="utf-8")  # 一次性读取整个停用词文件内容
    except FileNotFoundError:  # 文件不存在时不阻断程序，只是不做停用词过滤
        logger.warning("停用词文件不存在，跳过停用词过滤: %s", _STOPWORDS_FILE)  # 记录警告日志
        return set()  # 返回空集合，表示不过滤任何词
    return {w.strip() for w in text.splitlines() if w.strip()}  # 按行拆分并去除空白，过滤掉空行，得到停用词集合


if _USER_DICT_FILE.exists():  # 判断领域词典文件是否存在
    jieba.load_userdict(str(_USER_DICT_FILE))  # 存在则加载到 jieba，让分词优先识别校园相关专有名词
else:
    logger.warning("领域词典不存在: %s", _USER_DICT_FILE)  # 不存在则记录警告，仍可使用默认词典分词

_STOPWORDS = _load_stopwords()  # 模块加载时即读取停用词集合，供 tokenize() 复用


def tokenize(text: str) -> list[str]:
    """jieba 精确分词 + 去停用词。英文统一小写。"""
    if not text:  # 空字符串或 None 直接返回空列表，避免后续处理报错
        return []
    tokens = []  # 用于收集最终保留下来的分词结果
    for w in jieba.lcut(text.strip().lower()):  # 先去首尾空白并转小写（统一英文大小写），再用 jieba 精确模式分词
        w = w.strip()  # 去掉分出的词两侧可能残留的空白字符
        if not w or w in _STOPWORDS:  # 跳过空字符串和命中停用词表的词
            continue
        tokens.append(w)  # 保留有效词
    return tokens  # 返回分词结果列表


def best_match(query_tokens: list[str],
               tokenized_corpus: list[list[str]]) -> tuple[int, float]:
    """返回 (best_index, normalized_score)。语料或 query 为空时返回 (-1, 0.0)。

    best_index 是 tokenized_corpus 中的下标（不含临时追加的 query 文档）。
    """
    if not query_tokens or not tokenized_corpus:  # 查询分词为空 或 语料库为空时，无法匹配
        return -1, 0.0  # 直接返回"未命中"结果

    # query 自身作为临时文档追加到末尾
    bm25 = BM25Okapi(tokenized_corpus + [query_tokens])  # 用"语料 + query自身"构建 BM25 索引（追加文档法）
    scores = bm25.get_scores(query_tokens)  # 计算 query 相对语料中每篇文档（含自身）的 BM25 原始得分

    self_raw = float(scores[-1])  # 最后一项是 query 对自身的打分，作为归一化分母
    corpus_scores = scores[:-1]  # 除最后一项外，其余为 query 对语料库各文档的打分

    best_idx = int(max(range(len(corpus_scores)), key=lambda i: corpus_scores[i]))  # 找出得分最高的文档下标
    best_raw = float(corpus_scores[best_idx])  # 取出该最高分的原始数值

    if self_raw <= 0:  # 自打分非正时无法用作分母（避免除零或负数导致的异常归一化）
        logger.debug("BM25 self_raw=%.4f <= 0，归一化记 0", self_raw)  # 记录调试日志
        return best_idx, 0.0  # 归一化相似度记为 0

    norm = max(0.0, min(1.0, best_raw / self_raw))  # 用"最高分/自打分"做归一化，并截断到 [0, 1] 区间
    logger.debug("BM25 best_idx=%d raw=%.4f self=%.4f norm=%.4f",
                 best_idx, best_raw, self_raw, norm)  # 记录本次匹配的详细调试信息
    return best_idx, norm  # 返回最佳匹配下标和归一化后的相似度


if __name__ == "__main__":  # 直接运行本文件时执行的自测代码
    corpus = [tokenize(s) for s in [  # 对示例语料逐句分词，构建分词后的语料库
        "大一新生去哪里报到？",
        "报到需要带什么材料？",
        "图书馆开放时间？",
    ]]
    for probe in ["新生在哪儿报到", "报到要带哪些东西", "今天天气如何"]:  # 遍历几个测试查询
        idx, score = best_match(tokenize(probe), corpus)  # 对每个查询分词后与语料库做 BM25 匹配
        print(f"{probe!r} -> idx={idx} score={score:.4f} (阈值 {config.THRESHOLD})")  # 打印匹配结果及配置的阈值
