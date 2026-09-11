# core/strategy_selector.py
# 用于选择提示词策略: 直接检索策略，假设问题策略，子查询策略，回溯问题策略


# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate  # 用于创建带占位符的提示词模板
# 导入日志和配置
from base.config import Config  # 项目统一配置类
from base.logger import logger  # 项目统一日志对象
# 导入 OpenAI
from core.llm_client import LLMClient  # 封装好的大模型调用客户端

# todo 定义StrategySelector类: 用于根据用户查询选择最合适的检索增强策略.
class StrategySelector:
    # todo 1. 初始化方法
    def __init__(self):
        # 1. 加载配置
        self.conf = Config()  # 实例化配置对象
        # 2. 调用 OpenAI 客户端
        self.llm = LLMClient()  # ← 复用  # 复用已封装的 LLM 客户端，避免重复创建连接
        # 3. 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()  # 预先构建好策略选择用的 Prompt 模板，供后续填充

    # todo 2. 调用大模型的API -> 向DashScope发送请求, 获取模型返回结果.
    def call_dashscope(self, prompt):
        # 策略选择需要低温度
        result = self.llm.call(  # 调用大模型客户端发起请求
            prompt=prompt,  # 传入已经填充好占位符的完整提示词
            system_prompt="你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果",  # 系统角色设定，约束模型严格按指令输出
            temperature=0.1  # 低温度，让策略选择结果更确定、更少随机性
        )
        return result if result else "直接检索"  # 若调用失败返回空结果，则兜底使用"直接检索"策略

    # todo 3. 获取策略选择提示模板 -> 定义引导大模型选择策略的固定格式文本.
    @staticmethod
    def _get_strategy_prompt():
#   定义私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：新生报到需要带哪些材料？
                        * 策略：直接检索
                    * 查询：图书馆几点开门？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：作为大一新生，怎样才能更快适应大学生活？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：报到当天的流程、需要带的材料、以及宿舍怎么分配？
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我是外省考生，坐高铁到站后当天赶得及报到并领校园卡吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],  # 声明模板中出现的占位符名称
        )

    # todo 4. 定义方法，选择检索策略 -> 选择检索策略的核心方法 -> 整合模板和大模型调用, 返回最终策略.
    def select_strategy(self, query):
        """
        函数作用: 根据用户查询, 选择最合适的检索增强策略.
        :param query: 用户输入的查询文本(字符串)
        :return: 字符串 -> 选中的检索策略名称 -> 例如: 直接检索, 子查询检索...
        """
        # 1. 格式化提示模板: 将用户查询填充到提示模板的query为止, 生成发给大模型的完整提示, 调用大模型获取策略.
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()  # 填充模板并调用大模型，再去除首尾空白
        # 2. 记录日志.
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")  # 记录本次查询选中的策略
        # 3. 返回选中的策略.
        return strategy  # 返回策略名称字符串

if __name__ == '__main__':  # 直接运行本文件时执行的自测代码
    # 1. 实例化策略选择器
    ss = StrategySelector()  # 创建策略选择器实例
    # 2. 测试策略选择
    # ss.select_strategy('MySQL数据库能不能支持100W个样本的插入')  # 已注释的测试用例
    ss.select_strategy('对比北京和上海的人才引进政策, 从补贴金额, 落户难度, 产业适配性三个方面分析哪个更适合计算机专业毕业生')  # 执行一次策略选择测试
    # ss.select_strategy('如何培养孩子的时间管理能力')  # 已注释的测试用例