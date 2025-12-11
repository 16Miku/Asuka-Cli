"""
LLM API封装模块
支持OpenAI和Anthropic两种API
"""
import json
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

from config import config


@dataclass
class Message:
    """消息数据类"""
    role: str  # system, user, assistant, tool
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class LLMClient:
    """统一的LLM客户端"""
    
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or config.provider
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化对应的API客户端"""
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url
            )
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(
                api_key=config.anthropic_api_key
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}")
    
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            tools: 可用工具列表
            stream: 是否流式输出
            
        Returns:
            包含回复内容和工具调用的字典
        """
        if self.provider == "openai":
            return self._chat_openai(messages, tools, stream)
        else:
            return self._chat_anthropic(messages, tools, stream)
    
    def _chat_openai(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """OpenAI API调用"""
        # 转换消息格式
        formatted_messages = []
        for msg in messages:
            formatted_msg = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                formatted_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            formatted_messages.append(formatted_msg)
        
        # 构建请求参数
        kwargs = {
            "model": config.openai_model,
            "messages": formatted_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        if stream:
            return self._stream_openai(kwargs)
        
        # 非流式调用
        response = self._client.chat.completions.create(**kwargs)
        
        result = {
            "content": response.choices[0].message.content or "",
            "tool_calls": None,
            "finish_reason": response.choices[0].finish_reason
        }
        
        # 处理工具调用
        if response.choices[0].message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response.choices[0].message.tool_calls
            ]
        
        return result
    
    def _stream_openai(self, kwargs: Dict) -> Generator[Dict, None, None]:
        """OpenAI流式输出"""
        kwargs["stream"] = True
        stream = self._client.chat.completions.create(**kwargs)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"type": "content", "content": chunk.choices[0].delta.content}
            if chunk.choices[0].delta.tool_calls:
                yield {"type": "tool_call", "tool_calls": chunk.choices[0].delta.tool_calls}
    
    def _chat_anthropic(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Anthropic API调用"""
        # 提取system消息
        system_content = ""
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "tool":
                # Anthropic使用tool_result格式
                formatted_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content
                        }
                    ]
                })
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # 构建请求参数
        kwargs = {
            "model": config.anthropic_model,
            "max_tokens": config.max_tokens,
            "messages": formatted_messages,
        }
        
        if system_content:
            kwargs["system"] = system_content
        
        if tools:
            # 转换工具格式为Anthropic格式
            kwargs["tools"] = self._convert_tools_to_anthropic(tools)
        
        # 非流式调用
        response = self._client.messages.create(**kwargs)
        
        result = {
            "content": "",
            "tool_calls": None,
            "finish_reason": response.stop_reason
        }
        
        # 处理响应内容
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                if result["tool_calls"] is None:
                    result["tool_calls"] = []
                result["tool_calls"].append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        return result
    
    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """将OpenAI格式的工具定义转换为Anthropic格式"""
        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
        return anthropic_tools
