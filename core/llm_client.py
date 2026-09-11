# core/llm_client.py
from openai import OpenAI  # 导入 OpenAI 官方 SDK 的客户端类（兼容多数大模型 API）
from base.config import Config  # 导入项目统一的配置类，读取密钥、模型名等配置项
from base.logger import logger  # 导入项目统一的日志对象，用于记录调用异常

# LLMClient类实现方法
# 1. 初始化
# 2. 调用模型并得到响应的方法，参数prompt, system_prompt, temperature, stream
# 3. 流式调用方法
class LLMClient:  # 定义大语言模型（LLM）调用客户端类
    def __init__(self):
        self.config = Config()  # 实例化配置对象，读取 .env 等配置信息
        self.client = OpenAI(  # 创建 OpenAI 客户端实例，用于向大模型服务发起请求
            api_key=self.config.LLM_API_KEY,      # 从配置中读取 API 密钥
            base_url=self.config.LLM_BASE_URL     # 从配置中读取接口的基础 URL（支持第三方兼容服务）
        )

    def call(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, stream: bool = False):
        # prompt: 用户提问内容；system_prompt: 系统角色设定；temperature: 生成随机性；stream: 是否流式输出
        try:
            messages = [  # 构造符合 OpenAI Chat 接口规范的消息列表
                {"role": "system", "content": system_prompt or "你是一个有用的助手。"},  # 系统消息：若未传入则使用默认人设
                {"role": "user", "content": prompt},  # 用户消息：即传入的具体问题内容
            ]
            completion = self.client.chat.completions.create(  # 调用大模型对话补全接口
                model=self.config.LLM_MODEL,  # 指定使用的模型名称（来自配置）
                messages=messages,            # 传入上面构造好的消息列表
                temperature=temperature,      # 传入温度参数，控制回答的随机性/创造性
                stream=stream                 # 传入是否使用流式返回
            )
            if stream:
                # 流式模式：返回生成器（使用生成器函数）
                return self._stream_response(completion)  # 将原始流式响应对象交给生成器函数逐块处理
            else:
                # 非流式模式：直接返回字符串
                result = completion.choices[0].message.content  # 取出第一个候选回答的文本内容
                return result  # 直接返回完整的回答字符串
        except Exception as e:  # 捕获调用过程中可能出现的任何异常（网络、鉴权、限流等）
            logger.error(f"LLM 调用失败: {e}")  # 记录错误日志，便于排查问题
            return f"抱歉，处理您的问题时出现错误。请联系人工客服：{self.config.CUSTOMER_SERVICE_PHONE}"  # 向用户返回友好的兜底提示

    @staticmethod
    def _stream_response(completion):
        """生成器函数：逐词返回流式响应"""
        for chunk in completion:  # 遍历流式响应中的每一个数据块（chunk）
            if chunk.choices and chunk.choices[0].delta.content:  # 判断该块是否包含有效的增量文本内容
                yield chunk.choices[0].delta.content  # 通过 yield 逐块产出文本片段，实现流式输出