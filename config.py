"""
配置管理模块
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# 加载环境变量
load_dotenv()


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "openai"  # openai 或 anthropic
    
    # OpenAI配置
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    
    # Anthropic配置
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    
    # 通用配置
    max_tokens: int = 4096
    temperature: float = 0.7


def load_config() -> LLMConfig:
    """从环境变量加载配置"""
    return LLMConfig(
        provider=os.getenv("DEFAULT_LLM_PROVIDER", "openai"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    )


# 全局配置实例
config = load_config()
