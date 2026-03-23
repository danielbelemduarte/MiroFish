"""
LLM客户端封装
支持OpenAI和Azure OpenAI
"""

import json
import re
from typing import Optional, Dict, Any, List, Union
from openai import OpenAI, AzureOpenAI

from ..config import Config


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
) -> Union[OpenAI, AzureOpenAI]:
    """根据配置创建OpenAI或AzureOpenAI客户端"""
    if azure_endpoint or Config.use_azure_openai():
        return AzureOpenAI(
            api_key=api_key or Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=azure_endpoint or Config.AZURE_OPENAI_ENDPOINT,
            api_version=azure_api_version or Config.AZURE_OPENAI_API_VERSION,
        )
    else:
        return OpenAI(
            api_key=api_key or Config.LLM_API_KEY,
            base_url=base_url or Config.LLM_BASE_URL,
        )


def get_model_name(model: Optional[str] = None) -> str:
    """获取模型名称，Azure时使用deployment名称"""
    if model:
        return model
    if Config.use_azure_openai():
        return Config.AZURE_OPENAI_DEPLOYMENT or Config.LLM_MODEL_NAME
    return Config.LLM_MODEL_NAME


def _is_reasoning_model(model: str) -> bool:
    """检测是否为推理模型（不支持temperature和max_tokens）"""
    m = model.lower()
    return any(k in m for k in ("o1", "o3", "o4", "gpt-5"))


class LLMClient:
    """LLM客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.model = get_model_name(model)

        if Config.use_azure_openai():
            self.api_key = api_key or Config.AZURE_OPENAI_API_KEY
        else:
            self.api_key = api_key or Config.LLM_API_KEY

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.client = create_openai_client(
            api_key=api_key,
            base_url=base_url,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        reasoning = _is_reasoning_model(self.model)
        effective_max = max(max_tokens, 16384) if reasoning else max_tokens

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": effective_max,
        }

        if not reasoning:
            kwargs["temperature"] = temperature

        if response_format:
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")

