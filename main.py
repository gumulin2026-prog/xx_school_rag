# main.py
# xx_school_rag 迎新问答助手 —— 命令行交互入口

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import uuid

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from base.config import Config
from base.logger import logger
from core.mysql_client import MySQLClient
from core.rag_system import RAGSystem


def main():
    # 1. 初始化
    config = Config()
    mysql_client = MySQLClient()
    rag_system = RAGSystem()

    # 2. 生成会话 ID
    session_id = str(uuid.uuid4())

    # 3. 欢迎信息
    print("\n欢迎使用 XX 大学迎新问答助手！")
    print(f"会话 ID: {session_id}")
    print(f"支持的分类过滤：{config.VALID_SOURCES}")
    print("输入问题进行问答，输入 'exit' 退出，输入 'clear' 清除历史。")
    print("-" * 60)

    try:
        while True:
            # 4. 用户输入
            question = input("\n请输入问题: ").strip()
            if not question:
                continue
            if question.lower() == "exit":
                logger.info("退出系统")
                print("再见，祝你在 XX 大学一切顺利！")
                break
            if question.lower() == "clear":
                mysql_client.clear_session_history(session_id)
                print("历史已清除")
                continue

            # 5. 分类过滤
            source_filter = input(
                f"分类过滤 ({'/'.join(config.VALID_SOURCES)}, 回车跳过): "
            ).strip()
            if source_filter and source_filter not in config.VALID_SOURCES:
                print(f"无效分类 '{source_filter}'，跳过过滤")
                source_filter = None

            # 6. 取对话历史
            history = mysql_client.get_session_history(session_id)

            # 7. 生成答案（流式）
            print("\n助手: ", end="", flush=True)
            answer = ""
            for token in rag_system.generate_answer(
                query=question,
                source_filter=source_filter,
                history=history,
                stream=True,
            ):
                if token:
                    print(token, end="", flush=True)
                    answer += token
            print()

            # 8. 保存对话历史
            if answer and session_id:
                try:
                    mysql_client.update_session_history(session_id, question, answer)
                except Exception as e:
                    logger.error(f"保存历史失败: {e}")

    except KeyboardInterrupt:
        print("\n\n退出系统")
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"\n错误: {e}")
    finally:
        mysql_client.close()
        logger.info("系统退出")


if __name__ == '__main__':
    main()
