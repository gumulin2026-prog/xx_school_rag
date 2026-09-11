# xx_school_rag/core/mysql_client.py
# 该脚本用于: MySQL 访问层
#   - qa_knowledge  : 迎新问答知识库（question / answer / category）
#   - conversations : 多轮对话历史（session_id / question / answer / timestamp）

# 导包
import csv  # 用于读取 CSV 格式的知识库数据文件
import os   # 路径拼接与文件存在性判断
import sys  # 用于修改模块搜索路径 sys.path

import pymysql  # MySQL 数据库驱动

# 路径配置：把项目根目录加入 sys.path，便于以脚本方式直接运行本文件
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../xx_shcool_rag/core
project_root = os.path.dirname(current_dir)                 # .../xx_shcool_rag
sys.path.insert(0, project_root)  # 将项目根目录插入模块搜索路径最前面，保证 `from base import ...` 可用
from base import Config, logger  # 项目统一配置类和日志对象

# MySQLClient类提供的方法
# 1. 初始化方法
# 2. 创建数据表方法
# 3. 从 CSV 批量导入知识库（CSV 列: question, answer, category；GBK 编码）
# 4. 获取知识库全部问题（供 BM25 问题库初始化）
# 5. 按问题精确匹配查答案
# 6. 向知识库插入单条问答对（如需人工沉淀高频问答时调用）

# todo 定义 MySQLClient 类：连接管理、建表、导入数据、问答查询、对话历史
class MySQLClient:
    # todo 1. 初始化方法
    def __init__(self):
        self.logger = logger  # 保存日志对象为实例属性
        try:
            self.connection = pymysql.connect(  # 建立到 MySQL 服务器的连接
                host=Config().MYSQL_HOST,               # 数据库主机地址
                port=Config().MYSQL_PORT,               # 数据库端口
                user=Config().MYSQL_USER,               # 登录用户名
                password=str(Config().MYSQL_PASSWORD),  # 登录密码（强制转字符串，避免配置为数字类型时报错）
                database=Config().MYSQL_DATABASE,       # 目标数据库名
                charset="utf8mb4",                      # 字符集，支持中文及表情符号
            )
            self.cursor = self.connection.cursor()  # 创建游标对象，用于执行 SQL 语句
            self.logger.info("数据库连接成功")  # 记录连接成功日志
        except Exception as e:  # 捕获连接过程中可能出现的任何异常
            self.logger.error(f"数据库连接失败: {e}")  # 记录错误日志
            raise e  # 继续向上抛出异常，终止后续初始化

    # todo 2. 创建数据表方法
    def create_table(self):
        # 2.1 迎新问答知识库
        create_qa_knowledge = """
        CREATE TABLE IF NOT EXISTS qa_knowledge (
            id       INT AUTO_INCREMENT PRIMARY KEY,
            question VARCHAR(500) NOT NULL,
            answer   TEXT         NOT NULL,
            category VARCHAR(50)  DEFAULT NULL,
            UNIQUE KEY uk_question (question(255))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='迎新问答知识库'
        """
        # 2.2 多轮对话历史（一行一个问答对）
        create_conversations = """
        CREATE TABLE IF NOT EXISTS conversations (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(36) NOT NULL,
            question   TEXT        NOT NULL,
            answer     TEXT        NOT NULL,
            timestamp  DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session_id (session_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='多轮对话历史'
        """
        # 2.3 会话注册表：记录"哪些会话存在"，与会话是否已有消息解耦
        #     （没有这张表的话，刚创建、还没发过消息的新会话在 conversations 里没有任何行，
        #      会话列表接口就查不到它，导致新会话在切走后彻底找不回来）
        create_sessions = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR(36) PRIMARY KEY,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话注册表'
        """
        try:
            self.cursor.execute(create_qa_knowledge)  # 执行建表语句：不存在则创建 qa_knowledge 表
            self.cursor.execute(create_conversations)  # 执行建表语句：不存在则创建 conversations 表
            self.cursor.execute(create_sessions)  # 执行建表语句：不存在则创建 sessions 表
            # 兼容旧数据：把已经在 conversations 里出现过、但还没登记到 sessions 表的会话补录进来
            self.cursor.execute(
                "INSERT IGNORE INTO sessions (session_id, created_at) "
                "SELECT session_id, MIN(timestamp) FROM conversations GROUP BY session_id"
            )
            self.connection.commit()  # 提交事务，使建表和补录操作生效
            self.logger.info("数据表就绪: qa_knowledge / conversations / sessions")  # 记录成功日志
        except Exception as e:  # 捕获建表过程中可能出现的任何异常
            self.logger.error(f"创建表失败: {e}")  # 记录错误日志
            raise e  # 继续向上抛出异常

    # todo 3. 从 CSV 批量导入知识库（CSV 列: question, answer, category；GBK 编码）
    def insert_data(self, csv_path, encoding="gbk"):
        try:
            rows = []  # 用于收集待批量插入的 (question, answer, category) 元组
            with open(csv_path, encoding=encoding, newline="") as f:  # 按指定编码打开 CSV 文件
                for r in csv.DictReader(f):  # 以字典形式逐行读取（键为表头列名）
                    q = (r.get("question") or "").strip()  # 取问题列并去除首尾空白，缺失时用空字符串兜底
                    a = (r.get("answer") or "").strip()  # 取答案列并去除首尾空白
                    c = (r.get("category") or "").strip() or None  # 取分类列，去空白后若为空则记为 None
                    if q and a:  # 只有问题和答案都非空的行才有效
                        rows.append((q, a, c))  # 加入待插入列表
            self.cursor.executemany(  # 批量执行插入语句
                "INSERT INTO qa_knowledge (question, answer, category) "
                "VALUES (%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE answer = VALUES(answer), category = VALUES(category)",  # 若问题已存在（唯一键冲突）则更新答案和分类
                rows,
            )
            # ON DUPLICATE KEY UPDATE ：这是 MySQL 的特殊语法，意思是：如果插入时发现唯一键冲突（即这个问题已经存在了），就不报错，而是更新已有记录的字段。
            self.connection.commit()  # 提交事务
            self.logger.info(f"知识库导入成功: {len(rows)} 条")  # 记录成功导入的条数
        except Exception as e:  # 捕获导入过程中可能出现的任何异常
            self.logger.error(f"数据插入失败: {e}")  # 记录错误日志
            self.connection.rollback()  # 回滚事务，避免部分写入
            raise  # 继续向上抛出异常

    # todo 4. 获取知识库全部问题（供 BM25 问题库初始化）
    def fetch_questions(self):
        try:
            self.cursor.execute("SELECT question FROM qa_knowledge")  # 查询所有问题
            results = self.cursor.fetchall()  # 取出全部结果，格式为 [(问题,), ...]
            self.logger.info("数据查询成功")  # 记录成功日志
            return results  # 返回查询结果
        except pymysql.MySQLError as e:  # 捕获数据库相关异常
            self.logger.error(f"数据查询失败: {e}")  # 记录错误日志
            return []  # 出错时返回空列表，避免调用方崩溃

    # todo 5. 按问题精确匹配查答案
    def fetch_answer(self, question):
        try:
            self.cursor.execute(
                "SELECT answer FROM qa_knowledge WHERE question = %s LIMIT 1", (question,)  # 按问题原文精确匹配，只取一条
            )
            result = self.cursor.fetchone()  # 取出单条结果（元组或 None）
            return result[0] if result else None  # 命中则返回答案文本，否则返回 None
        except pymysql.MySQLError as e:  # 捕获数据库相关异常
            self.logger.error(f"数据查询失败: {e}")  # 记录错误日志
            return None  # 出错时返回 None

    # todo 6. 向知识库插入单条问答对（如需人工沉淀高频问答时调用）
    def insert_qa(self, question: str, answer: str, category: str = None):
        try:
            self.cursor.execute(
                "INSERT INTO qa_knowledge (question, answer, category) VALUES (%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE answer = VALUES(answer), category = VALUES(category)",  # 存在则更新，不存在则插入
                (question, answer, category),
            )
            self.connection.commit()  # 提交事务
            logger.info(f"问答对已加入知识库: {question[:30]}...")  # 记录日志（截取问题前30字符避免过长）
        except pymysql.MySQLError as e:  # 捕获数据库相关异常
            logger.error(f"插入知识库失败: {e}")  # 记录错误日志
            self.connection.rollback()  # 回滚事务
            raise  # 继续向上抛出异常

    # todo 7. 保存问答对到对话历史
    def save_conversation(self, session_id: str, question: str, answer: str):
        try:
            self.cursor.execute(
                "INSERT INTO conversations (session_id, question, answer, timestamp) "
                "VALUES (%s, %s, %s, NOW())",  # timestamp 使用数据库当前时间
                (session_id, question, answer),
            )
            self.connection.commit()  # 提交事务
            logger.info(f"会话 {session_id} 问答对保存成功")  # 记录成功日志
        except pymysql.MySQLError as e:  # 捕获数据库相关异常
            logger.error(f"保存问答对失败: {e}")  # 记录错误日志
            self.connection.rollback()  # 回滚事务
            raise  # 继续向上抛出异常

    # todo 8. 获取指定会话最近 N 轮对话历史（按时间正序返回）
    def get_session_history(self, session_id: str, limit: int = 5):
        try:
            self.cursor.execute(
                "SELECT question, answer FROM conversations WHERE session_id = %s "
                "ORDER BY timestamp DESC LIMIT %s",  # 按时间倒序取最近 limit 条
                (session_id, limit),
            )
            results = self.cursor.fetchall()  # 取出查询结果（倒序，最新的在前）
            return [{"question": row[0], "answer": row[1]} for row in results[::-1]]  # 反转为时间正序，并转换为字典列表
        except pymysql.MySQLError as e:  # 捕获数据库相关异常
            logger.error(f"获取会话历史失败: {e}")  # 记录错误日志
            return []  # 出错时返回空列表

    # 内部别名，保持与旧代码兼容
    _fetch_recent_history = get_session_history

    # todo 9. 更新会话：插入新记录，只保留最近 N 轮
    def update_session_history(self, session_id: str, question: str, answer: str, limit: int = 5):
        try:
            self.cursor.execute(
                "INSERT INTO conversations (session_id, question, answer, timestamp) "
                "VALUES (%s, %s, %s, NOW())",  # 先插入本轮新的问答记录
                (session_id, question, answer),
            )
            history = self.get_session_history(session_id, limit)  # 取出插入后最新的最近 limit 轮历史
            # 删除超出 limit 轮的旧记录
            self.cursor.execute(
                """
                DELETE FROM conversations
                WHERE session_id = %s
                  AND id NOT IN (
                      SELECT id FROM (
                          SELECT id FROM conversations
                          WHERE session_id = %s
                          ORDER BY timestamp DESC LIMIT %s
                      ) AS sub
                  )
                """,  # 子查询先选出最近 limit 条的 id，再删除该会话中不在这些 id 内的所有旧记录（子查询需嵌套 AS sub 以规避 MySQL 不能直接对同表 DELETE + 子查询的限制）
                (session_id, session_id, limit),
            )
            self.connection.commit()  # 提交事务（插入 + 清理旧记录一起生效）
            logger.info(f"会话 {session_id} 历史更新成功，保留最近 {len(history)} 轮")  # 记录成功日志
            return history  # 返回更新后的历史记录列表
        except Exception as e:  # 捕获过程中可能出现的任何异常
            logger.error(f"更新会话历史失败: {e}")  # 记录错误日志
            self.connection.rollback()  # 回滚事务
            raise  # 继续向上抛出异常

    # todo 10. 清除会话历史
    def clear_session_history(self, session_id: str):
        try:
            self.cursor.execute(
                "DELETE FROM conversations WHERE session_id = %s", (session_id,)  # 删除该会话的全部历史记录
            )
            self.connection.commit()  # 提交事务
            logger.info(f"会话 {session_id} 历史已清除")  # 记录成功日志
        except pymysql.MySQLError as e:  # 捕获数据库相关异常
            logger.error(f"清除会话历史失败: {e}")  # 记录错误日志
            self.connection.rollback()  # 回滚事务
            raise  # 继续向上抛出异常

    # todo 11. 关闭连接
    def close(self):
        try:
            self.connection.close()  # 关闭数据库连接，释放资源
            self.logger.info("MySQL 连接已关闭")  # 记录关闭成功日志
        except pymysql.MySQLError as e:  # 捕获关闭过程中可能出现的异常
            self.logger.error(f"关闭连接失败: {e}")  # 记录错误日志（此处未抛出，避免影响调用方收尾流程）


if __name__ == '__main__':  # 直接运行本文件时执行的自测代码
    client = MySQLClient()  # 创建客户端实例，建立数据库连接
    client.create_table()  # 确保所需的两张表已创建
    csv_path = os.path.join(project_root, "data", "mysql_data", "new_student_qa.csv")  # 待导入的知识库 CSV 文件路径
    client.insert_data(csv_path)  # 批量导入知识库数据
    print("知识库问题数:", len(client.fetch_questions()))  # 打印当前知识库问题总数
    print("示例:", client.fetch_answer("大一新生去哪里报到？"))  # 打印一条示例问题的答案
    client.close()  # 关闭数据库连接
