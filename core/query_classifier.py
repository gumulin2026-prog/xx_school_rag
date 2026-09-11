# \core\query_classifier.py
# 此模块用于分析用户的问题进行意图识别: 判断用户的问题是专业咨询还是通用问题

# 导包
# 1. 导入标准库
import json, os, sys  # json: 解析训练数据文件; os: 路径与文件操作; sys: 未直接使用但保留以备扩展

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件（query_classifier.py）所在目录的绝对路径
project_root = os.path.dirname(current_dir)  # 项目根目录 = core 目录的上一级

# 2. 导入项目模块
from base.logger import logger  # 项目统一日志对象
# 3. 导入pytorch库
import torch  # 深度学习框架，用于模型推理、张量运算、设备管理
# 4. 导入numpy库
import numpy as np  # 数值计算库，用于处理 logits、计算准确率等
# 5. 导入transformers库(负责模型训练)，Trainer具体执行，TrainingArguments设置训练参数: 训练轮数，批次数...
from transformers import Trainer, TrainingArguments  # Trainer: 封装训练/评估循环; TrainingArguments: 训练超参数配置
from transformers import BertTokenizer, BertForSequenceClassification  # BERT 分词器 与 序列分类模型
# 6. 导入机器学习库
from sklearn.model_selection import train_test_split  # 用于划分训练集/验证集
from sklearn.metrics import classification_report, confusion_matrix  # 用于生成分类报告和混淆矩阵


# todo 1. 定义QueryClassifier类: 封装BERT查询分类的完整流程: 模型加载 → 训练 → 评估 → 预测 → ...
class QueryClassifier:
    # todo 1.1 初始化方法
    def __init__(self, model_path=None):
        # 1. 存储模型路径: 用于后续加载或保存模型.
        # 如果未指定模型路径，使用默认路径
        if model_path is None:  # 未显式指定模型路径时
            model_path = os.path.join(project_root, 'models', 'bert_query_classifier')  # 使用项目内默认的模型保存目录
        self.model_path = model_path  # 保存为实例属性，供 load_model / save_model 使用
        # 2. 加载BERT分词器: 将文本转换成模型可以理解的输入
        self.tokenizer = None  # 先占位为 None，实际对象由 load_model() 赋值
        # 3. 初始化模型变量，后续通过load_model()加载或创建模型
        self.model = None  # 先占位为 None，实际对象由 load_model() 赋值
        # 4. 选择模型运行设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有可用 GPU 则用 cuda，否则退回 cpu
        logger.info(f'使用设备: {self.device}')  # 记录本次运行使用的设备
        # 5. 定义标签映射
        self.load_map = {"通用知识": 0, "专业咨询": 1}  # 文本标签到数字标签的映射，供训练数据编码使用
        # 6. 加载模型: 初始化时，自动调用load_model()，确保模型可用
        self.load_model()  # 构造时立即加载/初始化模型，保证实例可用

    # todo 1.2 加载模型方法 → 从指定路径加载已训练模型，若不存在则初始化新模型
    def load_model(self):
        pretrained_bert_path = os.path.join(project_root, 'models', 'bert-base-chinese')  # 未微调过的原始中文 BERT 预训练模型路径
        if os.path.exists(self.model_path):  # 判断是否已存在训练好并保存过的模型目录
            # 目录存在：加载已保存的（包含微调权重的）全套模型
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)  # 从已保存目录加载分词器
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)  # 从已保存目录加载微调后的分类模型
            logger.info("从模型目录加载已保存的分词器和模型")  # 记录加载来源
        else:
            # 目录不存在：初始化全新的预训练模型
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_path)  # 从原始 BERT 路径加载分词器
            self.model = BertForSequenceClassification.from_pretrained(pretrained_bert_path, num_labels=2)  # 加载原始 BERT 并初始化二分类头
            logger.info("从原始 BERT 路径加载分词器并初始化新模型")  # 记录加载来源
        self.model.to(self.device)  # 将模型参数迁移到之前选定的运行设备（GPU 或 CPU）
        logger.info(f'使用设备: {self.device}')  # 再次记录当前使用的设备（模型迁移完成后确认）

    # todo 1.3. 保存模型的方法 -> 将训练好的模型和分词器存储到本地.
    def save_model(self):
        self.model.save_pretrained(self.model_path)  # 将模型权重和配置保存到指定目录
        self.tokenizer.save_pretrained(self.model_path)  # 将分词器的词表和配置保存到同一目录
        logger.info(f"保存模型至: {self.model_path}")  # 记录保存路径

    #  todo 1.4 数据预处理方法 -> 将文本和标签转换为BERT模型所需的输入格式.
    def preprocess_data(self, texts, labels):
        # 1. 创建分词器对象，并设置参数
        encodings = self.tokenizer(  # 调用分词器，将文本列表批量转换为模型输入张量
            texts,
            max_length=32,   # 迎新问句普遍较短，32 足够，10 会把长问句截断过狠
            padding=True,  # 是否填充  # 对齐批次内长度不足的样本，用 0 填充到相同长度
            truncation=True,  # 是否截断  # 超过 max_length 的样本从末尾截断
            return_tensors='pt'  # 返回张量  # 返回 PyTorch 张量（'pt' = pytorch）
        )
        # 2. 标签转换
        labels = [self.load_map[label] for label in labels]  # 将文本标签（如"通用知识"）转换为数字标签（如 0）
        # 3. 返回分词器对象和标签
        return encodings, labels  # 返回编码结果和数字化标签列表

    # todo 1.5. 创建数据集方法: 自定义Dataset类, 封装: 编码 和 标签.
    def create_dataset(self, encodings, labels):
        # 自定义数据集类
        class Dataset(torch.utils.data.Dataset):  # 继承 PyTorch Dataset 基类，供 Trainer 使用
            # 1. 初始化
            def __init__(self, encodings, labels):
                super().__init__()  # 调用父类构造方法
                self.encodings = encodings  # 文本编码, 格式: (input_ids, attention_mask)
                self.labels = labels  # 数字标签列表, 格式: [0, 1, 0, 1, ...]

            # 2. 获取指定索引的编码和标签.
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in
                        self.encodings.items()}  # self.encodings.items() 会返回 [('input_ids', [[...], [...]]), ('attention_mask', [[...], [...]])]  # 取出第 idx 个样本各字段的值，组装成字典
                item['labels'] = torch.tensor(self.labels[idx])  # 将该样本对应的数字标签转换为张量并加入字典
                return item  # 返回单个样本（含 input_ids, attention_mask, labels）

            # 3. 获取数据集长度
            def __len__(self):
                return len(self.labels)  # 数据集长度等于标签列表长度

        # 4. 返回实例化的Dataset对象.
        return Dataset(encodings, labels)  # 用传入的编码和标签构造并返回 Dataset 实例

    # todo 1.6 训练模型方法 -> 加载数据集, 预处理, 配置训练参数并训练模型.
    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
        # 1. 检查数据集文件是否存在
        if not os.path.exists(data_file):  # 数据集文件不存在则无法训练
            logger.info(f"数据集文件 {data_file} 不存在")  # 记录提示日志
            raise FileExistsError(f"数据集文件 {data_file} 不存在")  # 抛出异常终止训练流程

        # 2. 加载数据集, 从Json文件中读取查询文本和对应标签.
        with open(data_file, 'r', encoding='utf-8') as f:  # 以 UTF-8 编码打开数据集文件
            data = [json.loads(value) for value in f.readlines()]  # 逐行读取并解析为 JSON 对象（JSONL 格式）

        # 3. 提取文本和标签，分离查询文本和对应的分类标签.
        texts = [item["query"] for item in data]  # 提取每条样本的查询文本
        labels = [item["label"] for item in data]  # 提取每条样本对应的标签

        # 4. 划分训练集和测试集. 80%用于训练, 20%用于验证. 固定随机种子确保结果可复现.
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2,
                                                                            random_state=42)  # 按 8:2 划分训练/验证集，固定随机种子保证可复现

        # 5. 数据预处理: 将文本和标签转换为BERT模型所需的输入格式 -> 调用自定义的preprocess_data() 预处理数据的方法.
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)  # 训练集文本编码 + 标签数字化
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)  # 验证集文本编码 + 标签数字化

        # 6. 构建数据集: 将编码 和 标签封装为PyTorch可识别的 Dataset对象.
        train_dataset = self.create_dataset(train_encodings, train_labels)  # 封装为训练用 Dataset
        val_dataset = self.create_dataset(val_encodings, val_labels)  # 封装为验证用 Dataset

        # 7. 配置训练参数: 定义模型训练的超参数.
        training_args = TrainingArguments(
            # 设置模型和检查点保存的目录路径
            output_dir=os.path.join(project_root, "bert_results"),
            # 设置训练的总轮数为3轮
            num_train_epochs=3,
            # 设置每个设备（GPU/CPU）上的训练批次大小为8
            per_device_train_batch_size=8,
            # 设置每个设备（GPU/CPU）上的评估批次大小为8
            per_device_eval_batch_size=8,
            # 设置学习率预热步数为20步，训练初期学习率从0逐渐增加到设定值, 逐步提升学习率, 避免初始震荡.
            warmup_steps=20,
            # 设置权重衰减系数为0.01，用于防止过拟合
            weight_decay=0.01,
            # 设置每10个训练步骤记录一次日志
            logging_steps=10,
            # 不上报到 wandb / tensorboard
            report_to="none",
            # 设置评估策略为每个epoch结束后进行评估（transformers 5.x 参数名为 eval_strategy）
            eval_strategy="epoch",
            # 设置模型保存策略为每个epoch结束后保存
            save_strategy="epoch",
            # 设置训练结束后加载最佳模型而非最后一个模型
            load_best_model_at_end=True,
            # 设置最多保存1个检查点文件，超出时自动删除旧的
            save_total_limit=1,
            # 设置用于判断最佳模型的指标为评估损失
            metric_for_best_model="eval_loss",
            # 禁用FP16混合精度训练，使用FP32精度 简化配置, 需要GPU支持
            fp16=False,
        )

        # 8. 初始化 Trainer: 封装模型, 训练参数, 数据集, 评估指标计算方法.
        trainer = Trainer(
            # 传入要训练的模型实例
            model=self.model,
            # 传入上面定义的训练参数配置
            args=training_args,
            # 传入训练数据集
            train_dataset=train_dataset,
            # 传入验证数据集，用于训练过程中评估模型性能
            eval_dataset=val_dataset,
            # 传入计算评估指标的函数，用于在验证集上计算准确率等指标
            compute_metrics=self.compute_metrics  # 自定义评估指标(准确率)
        )

        # 9. 训练模型
        logger.info("开始训练 BERT 模型...")  # 记录训练开始日志
        trainer.train()  # 启动训练循环（含前向、反向传播、参数更新、按 epoch 评估）
        self.save_model()  # 训练结束后保存最优模型和分词器到本地

        # 10. 评估模型
        self.evaluate_model(val_texts, val_labels)  # 在验证集上生成分类报告和混淆矩阵，评估最终效果

    # todo 1.7 计算评估指标方法 -> 计算模型在验证集上的准确率.
    @staticmethod
    def compute_metrics(eval_pred):
        """
        函数功能: 计算模型在验证集上的准确率.
        :param eval_pred: 模型预测结果和标签.
        :return: 准确率.
        """
        # 1. 模型输出的 logits 和 真实标签.
        logits, labels = eval_pred  # 解包 Trainer 传入的 (模型原始输出, 真实标签) 元组
        # 2. 取logits最大值对应的索引作为预测结果.
        predictions = np.argmax(logits, axis=-1)  # 沿最后一维取最大值下标，即预测的类别编号
        # 3. 计算准确率: 预测正确的样本占比
        accuracy = (predictions == labels).mean()  # (predictions == labels)逐元素对比，返回一个布尔值列表  # 求布尔列表均值即为准确率
        # 4. 返回准确率字典.
        return {"accuracy": accuracy}  # Trainer 要求以字典形式返回各项指标

    # todo 1.8 评估模型方法: 生成分类报告和混淆矩阵, 全面分析模型性能.
    def evaluate_model(self, texts, labels):
        """
        函数功能: 在给定文本和标签上评估模型, 输出分类报告和混淆矩阵.
        :param texts:  待评估的文本列表
        :param labels: 文本对应的真实标签(数字形式)
        """
        # 1. 对文本进行编码，仅处理文本，因为标签已经转为数字了
        encodings = self.tokenizer(  # 用分词器批量编码待评估文本
            texts,
            truncation=True,  # 超长文本截断
            padding=True,  # 短文本填充对齐
            max_length=128,  # 评估时使用更宽松的最大长度
            return_tensors="pt"  # 返回 PyTorch 张量
        )

        # 2. 创建评估数据集
        dataset = self.create_dataset(encodings, labels)  # 封装为 Dataset 对象供 Trainer 使用

        # 3. 使用 Trainer 进行预测
        trainer = Trainer(model=self.model)  # 用当前已加载的模型创建一个仅用于预测的 Trainer
        predictions = trainer.predict(dataset)  # 对数据集执行推理，得到包含 logits 的预测结果

        # 4. 解析预测结果: 取logits最大值对应的索引作为预测结果.
        pred_labels = np.argmax(predictions.predictions, axis=-1)  # 取每条样本 logits 最大值对应的类别下标
        true_labels = labels  # 直接使用数字标签

        # 5. 记录评估结果到日志.
        logger.info("分类报告:")  # 打印分类报告小标题
        logger.info(classification_report(  # 生成包含精确率、召回率、F1 等指标的分类报告
            true_labels,
            pred_labels,
            target_names=["通用知识", "专业咨询"]  # 指定标签对应的可读名称
        ))

        # 6. 混淆矩阵: 显示预测标签和真实标签的匹配情况.
        logger.info("混淆矩阵:")  # 打印混淆矩阵小标题
        logger.info(confusion_matrix(true_labels, pred_labels))  # 计算并打印混淆矩阵

    # todo 1.9 预测类别方法 -> 对单个查询文本进行分类预测.
    def predict_category(self, query):
        """
        函数功能: 对单个查询文本进行类别预测 -> 通用知识 或者 专业咨询.
        :param query: 待分类的查询文本
        :return: 类别名称(文本标签)
        """
        # 1. 检查模型是否加载.
        if self.model is None:  # 防御性检查：理论上 __init__ 已保证模型已加载
            # 模型未加载
            logger.error("模型未训练或加载!")  # 记录错误日志
            return '通用知识'  # 兜底返回默认类别

        # 2. 对查询文本进行编码.
        encoding = self.tokenizer(  # 对单条查询文本编码
            query,
            truncation=True,  # 超长截断
            padding=True,  # 填充对齐
            max_length=128,  # 最大长度
            return_tensors="pt"  # 返回 PyTorch 张量
        )

        # 3. 将编码数据迁移到指定设备.
        encoding = {k: v.to(self.device) for k, v in encoding.items()}  # 把 input_ids、attention_mask 等张量搬到 GPU/CPU

        # 4. 模型推理 -> 不计算梯度, 提高效率.
        with torch.no_grad():  # 推理阶段关闭梯度计算，节省显存并加速
            # 4.1 获取模型输出 -> 包含logits
            outputs = self.model(**encoding)  # 前向传播，得到模型输出（含未归一化的 logits）
            # 4.2 获取预测结果 -> 获取logits最大值对应的索引作为预测结果.
            prediction = torch.argmax(outputs.logits, dim=1).item()  # 取 logits 最大值下标并转为 Python 整数

        # 5. 返回预测结果标签
        return '专业咨询' if prediction == 1 else '通用知识'  # 根据数字标签映射回可读的中文类别名


if __name__ == '__main__':  # 直接运行本文件时执行的训练与自测流程
    # 1. 实例化查询分类器 -> 自动加载模型, 若无则初始化新模型.
    query_classify = QueryClassifier()  # 创建分类器实例（构造时会自动加载/初始化模型）
    # 2. 训练模型（迎新意图数据集，先跑 scripts/gen_intent_dataset.py 生成）
    data_file = os.path.join(project_root, "data", "train_data", "classify_data", "new_student_intent.jsonl")  # 训练数据文件路径
    query_classify.train_model(data_file)  # 执行训练、保存模型、并在验证集上评估

    # 3. 示例预测
    for q in ["大一新生去哪里报到", "你好", "食堂能用微信吗", "1加1等于几"]:  # 遍历几个示例问题
        print(q, "->", query_classify.predict_category(q))  # 打印每个问题的预测类别
