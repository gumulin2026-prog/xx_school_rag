# 本地模型下载说明

本目录下的模型文件较大（合计约 6 GB），**不纳入 Git 版本管理**（见根目录 `.gitignore`）。
克隆项目后需自行下载到对应子目录。

| 子目录 | 用途 | 来源（任选其一） |
|---|---|---|
| `bge-m3/` | RAG 稠密+稀疏混合向量（`vector_store.py`） | ModelScope: `BAAI/bge-m3` · HuggingFace: `BAAI/bge-m3` |
| `bge-reranker-large/` | RAG 检索结果重排（`vector_store.py`） | ModelScope: `BAAI/bge-reranker-large` · HuggingFace: `BAAI/bge-reranker-large` |
| `bert-base-chinese/` | 意图分类器的基座模型（`query_classifier.py` 训练时用） | HuggingFace: `google-bert/bert-base-chinese` |
| `bert_query_classifier/` | 微调后的意图分类器（通用知识 / 专业咨询）。可用 `python -m core.query_classifier` 自行训练生成 | 由 `scripts/gen_intent_dataset.py` + `core/query_classifier.py` 训练产出 |
| `nlp_bert_document-segmentation_chinese-base/` | 达摩院语义分段模型（`AliTextSplitter`，当前默认切分器未启用，可选） | ModelScope: `damo/nlp_bert_document-segmentation_chinese-base` |

## 下载示例（ModelScope）

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-reranker-large', local_dir='models/bge-reranker-large')"
```

## 意图分类器

`bert_query_classifier/` 不必下载，本地训练即可：

```bash
python scripts/gen_intent_dataset.py --llm      # 生成训练集
python -m core.query_classifier                 # 训练并保存到 models/bert_query_classifier/
```
