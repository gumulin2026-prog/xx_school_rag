# xx_school_rag/core/mysql_client.py
# 该脚本用于: MySQL 访问层
#   - qa_knowledge  : 迎新问答知识库（question / answer / category）
#   - conversations : 多轮对话历史（session_id / question / answer / timestamp）

# 导包
import csv
import os
import sys

import pymysql

# 路径配置：把项目根目录加入 sys.path，便于以脚本方式直接运行本文件
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../xx_shcool_rag/core
project_root = os.path.dirname(current_dir)                 # .../xx_shcool_rag
sys.path.insert(0, project_root)
from base import Config, logger


# todo 定义 MySQLClient 类：连接管理、建表、导入数据、问答查询、对话历史
class MySQLClient:
    # todo 1. 初始化方法
    def __init__(self):
        self.logger = logger
        try:
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                port=Config().MYSQL_PORT,
                user=Config().MYSQL_USER,
                password=str(Config().MYSQL_PASSWORD),
                database=Config().MYSQL_DATABASE,
                charset="utf8mb4",
            )
            self.cursor = self.connection.cursor()
            self.logger.info("数据库连接成功")
        except Exception as e:
            self.logger.error(f"数据库连接失败: {e}")
            raise e

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
        try:
            self.cursor.execute(create_qa_knowledge)
            self.cursor.execute(create_conversations)
            self.connection.commit()
            self.logger.info("数据表就绪: qa_knowledge / conversations")
        except Exception as e:
            self.logger.error(f"创建表失败: {e}")
            raise e

    # todo 3. 从 CSV 批量导入知识库（CSV 列: question, answer, category；GBK 编码）
    def insert_data(self, csv_path, encoding="gbk"):
        try:
            rows = []
            with open(csv_path, encoding=encoding, newline="") as f:
                for r in csv.DictReader(f):
                    q = (r.get("question") or "").strip()
                    a = (r.get("answer") or "").strip()
                    c = (r.get("category") or "").strip() or None
                    if q and a:
                        rows.append((q, a, c))
            self.cursor.executemany(
                "INSERT INTO qa_knowledge (question, answer, category) "
                "VALUES (%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE answer = VALUES(answer), category = VALUES(category)",
                rows,
            )
            self.connection.commit()
            self.logger.info(f"知识库导入成功: {len(rows)} 条")
        except Exception as e:
            self.logger.error(f"数据插入失败: {e}")
            self.connection.rollback()
            raise

    # todo 4. 获取知识库全部问题（供 BM25 问题库初始化）
    def fetch_questions(self):
        try:
            self.cursor.execute("SELECT question FROM qa_knowledge")
            results = self.cursor.fetchall()
            self.logger.info("数据查询成功")
            return results
        except pymysql.MySQLError as e:
            self.logger.error(f"数据查询失败: {e}")
            return []

    # todo 5. 按问题精确匹配查答案
    def fetch_answer(self, question):
        try:
            self.cursor.execute(
                "SELECT answer FROM qa_knowledge WHERE question = %s LIMIT 1", (question,)
            )
            result = self.cursor.fetchone()
            return result[0] if result else None
        except pymysql.MySQLError as e:
            self.logger.error(f"数据查询失败: {e}")
            return None

    # todo 6. 向知识库插入单条问答对（如需人工沉淀高频问答时调用）
    def insert_qa(self, question: str, answer: str, category: str = None):
        try:
            self.cursor.execute(
                "INSERT INTO qa_knowledge (question, answer, category) VALUES (%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE answer = VALUES(answer), category = VALUES(category)",
                (question, answer, category),
            )
            self.connection.commit()
            logger.info(f"问答对已加入知识库: {question[:30]}...")
        except pymysql.MySQLError as e:
            logger.error(f"插入知识库失败: {e}")
            self.connection.rollback()
            raise

    # todo 7. 保存问答对到对话历史
    def save_conversation(self, session_id: str, question: str, answer: str):
        try:
            self.cursor.execute(
                "INSERT INTO conversations (session_id, question, answer, timestamp) "
                "VALUES (%s, %s, %s, NOW())",
                (session_id, question, answer),
            )
            self.connection.commit()
            logger.info(f"会话 {session_id} 问答对保存成功")
        except pymysql.MySQLError as e:
            logger.error(f"保存问答对失败: {e}")
            self.connection.rollback()
            raise

    # todo 8. 获取指定会话最近 N 轮对话历史（按时间正序返回）
    def get_session_history(self, session_id: str, limit: int = 5):
        try:
            self.cursor.execute(
                "SELECT question, answer FROM conversations WHERE session_id = %s "
                "ORDER BY timestamp DESC LIMIT %s",
                (session_id, limit),
            )
            results = self.cursor.fetchall()
            return [{"question": row[0], "answer": row[1]} for row in results[::-1]]
        except pymysql.MySQLError as e:
            logger.error(f"获取会话历史失败: {e}")
            return []

    # 内部别名，保持与旧代码兼容
    _fetch_recent_history = get_session_history

    # todo 9. 更新会话：插入新记录，只保留最近 N 轮
    def update_session_history(self, session_id: str, question: str, answer: str, limit: int = 5):
        try:
            self.cursor.execute(
                "INSERT INTO conversations (session_id, question, answer, timestamp) "
                "VALUES (%s, %s, %s, NOW())",
                (session_id, question, answer),
            )
            history = self.get_session_history(session_id, limit)
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
                """,
                (session_id, session_id, limit),
            )
            self.connection.commit()
            logger.info(f"会话 {session_id} 历史更新成功，保留最近 {len(history)} 轮")
            return history
        except Exception as e:
            logger.error(f"更新会话历史失败: {e}")
            self.connection.rollback()
            raise

    # todo 10. 清除会话历史
    def clear_session_history(self, session_id: str):
        try:
            self.cursor.execute(
                "DELETE FROM conversations WHERE session_id = %s", (session_id,)
            )
            self.connection.commit()
            logger.info(f"会话 {session_id} 历史已清除")
        except pymysql.MySQLError as e:
            logger.error(f"清除会话历史失败: {e}")
            self.connection.rollback()
            raise

    # todo 11. 关闭连接
    def close(self):
        try:
            self.connection.close()
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            self.logger.error(f"关闭连接失败: {e}")


if __name__ == '__main__':
    client = MySQLClient()
    client.create_table()
    csv_path = os.path.join(project_root, "data", "mysql_data", "new_student_qa.csv")
    client.insert_data(csv_path)
    print("知识库问题数:", len(client.fetch_questions()))
    print("示例:", client.fetch_answer("大一新生去哪里报到？"))
    client.close()
