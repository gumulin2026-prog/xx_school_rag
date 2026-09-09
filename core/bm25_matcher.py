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

import logging
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from base.config import config
from base.logger import logger

jieba.setLogLevel(logging.WARNING)  # 静默 "Building prefix dict..." 等启动噪声

_DATA_DIR = Path(config.DATA_DIR)
_STOPWORDS_FILE = _DATA_DIR / "stopwords.txt"
_USER_DICT_FILE = _DATA_DIR / "user_dict.txt"


def _load_stopwords() -> set[str]:
    try:
        text = _STOPWORDS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("停用词文件不存在，跳过停用词过滤: %s", _STOPWORDS_FILE)
        return set()
    return {w.strip() for w in text.splitlines() if w.strip()}


if _USER_DICT_FILE.exists():
    jieba.load_userdict(str(_USER_DICT_FILE))
else:
    logger.warning("领域词典不存在: %s", _USER_DICT_FILE)

_STOPWORDS = _load_stopwords()


def tokenize(text: str) -> list[str]:
    """jieba 精确分词 + 去停用词。英文统一小写。"""
    if not text:
        return []
    tokens = []
    for w in jieba.lcut(text.strip().lower()):
        w = w.strip()
        if not w or w in _STOPWORDS:
            continue
        tokens.append(w)
    return tokens


def best_match(query_tokens: list[str],
               tokenized_corpus: list[list[str]]) -> tuple[int, float]:
    """返回 (best_index, normalized_score)。语料或 query 为空时返回 (-1, 0.0)。

    best_index 是 tokenized_corpus 中的下标（不含临时追加的 query 文档）。
    """
    if not query_tokens or not tokenized_corpus:
        return -1, 0.0

    # query 自身作为临时文档追加到末尾
    bm25 = BM25Okapi(tokenized_corpus + [query_tokens])
    scores = bm25.get_scores(query_tokens)

    self_raw = float(scores[-1])
    corpus_scores = scores[:-1]

    best_idx = int(max(range(len(corpus_scores)), key=lambda i: corpus_scores[i]))
    best_raw = float(corpus_scores[best_idx])

    if self_raw <= 0:
        logger.debug("BM25 self_raw=%.4f <= 0，归一化记 0", self_raw)
        return best_idx, 0.0

    norm = max(0.0, min(1.0, best_raw / self_raw))
    logger.debug("BM25 best_idx=%d raw=%.4f self=%.4f norm=%.4f",
                 best_idx, best_raw, self_raw, norm)
    return best_idx, norm


if __name__ == "__main__":
    corpus = [tokenize(s) for s in [
        "大一新生去哪里报到？",
        "报到需要带什么材料？",
        "图书馆开放时间？",
    ]]
    for probe in ["新生在哪儿报到", "报到要带哪些东西", "今天天气如何"]:
        idx, score = best_match(tokenize(probe), corpus)
        print(f"{probe!r} -> idx={idx} score={score:.4f} (阈值 {config.THRESHOLD})")
