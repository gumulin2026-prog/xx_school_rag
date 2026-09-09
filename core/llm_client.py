# core/llm_client.py
from openai import OpenAI
from base.config import Config
from base.logger import logger

class LLMClient:
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL
        )

    def call(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, stream: bool = False):
        try:
            messages = [
                {"role": "system", "content": system_prompt or "你是一个有用的助手。"},
                {"role": "user", "content": prompt},
            ]
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                stream=stream
            )
            if stream:
                # 流式模式：返回生成器（使用生成器函数）
                return self._stream_response(completion)
            else:
                # 非流式模式：直接返回字符串
                result = completion.choices[0].message.content
                return result
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return f"抱歉，处理您的问题时出现错误。请联系人工客服：{self.config.CUSTOMER_SERVICE_PHONE}"

    @staticmethod
    def _stream_response(completion):
        """生成器函数：逐词返回流式响应"""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content