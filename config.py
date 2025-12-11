"""
配置管理模块

这是一个多行文档字符串（docstring），用于描述整个模块的用途。
本模块负责：
1. 从 .env 文件加载环境变量（如 API 密钥）
2. 定义配置数据结构（LLMConfig 类）
3. 提供全局配置实例供其他模块使用

为什么需要配置管理？
- 敏感信息（如 API 密钥）不应该硬编码在代码中
- 配置集中管理，便于修改和维护
- 支持不同环境（开发、测试、生产）使用不同配置
"""

# =====================================================================
# 导入部分
# =====================================================================

import os
# os 是 Python 标准库，提供与操作系统交互的功能
# 这里主要使用 os.getenv() 函数来读取环境变量
# 
# 什么是环境变量？
# - 操作系统级别的键值对配置
# - 可以在不修改代码的情况下改变程序行为
# - 常用于存储敏感信息（密码、API密钥等）
# 
# 例如：
# - Windows: set OPENAI_API_KEY=sk-xxx
# - Linux/Mac: export OPENAI_API_KEY=sk-xxx

from dotenv import load_dotenv
# dotenv 是第三方库（python-dotenv），需要 pip install python-dotenv
# 它的作用是从 .env 文件中读取配置，并设置为环境变量
# 
# .env 文件是什么？
# - 一个纯文本文件，存放在项目根目录
# - 格式：KEY=VALUE，每行一个配置
# - 通常被 .gitignore 忽略，不提交到代码仓库
# - 用于存储本地开发环境的敏感配置
#
# .env 文件示例：
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
# OPENAI_BASE_URL=https://api.openai.com/v1

from dataclasses import dataclass
# dataclass 是 Python 3.7+ 引入的装饰器，用于简化数据类的定义
# 
# 什么是数据类？
# - 主要用于存储数据的类
# - 自动生成 __init__、__repr__、__eq__ 等方法
# - 减少样板代码（boilerplate code）
#
# 不使用 dataclass 时，需要手动写：
# class LLMConfig:
#     def __init__(self, provider, openai_api_key, ...):
#         self.provider = provider
#         self.openai_api_key = openai_api_key
#         ...
#
# 使用 dataclass 后，只需声明属性，Python 自动生成上述代码

from typing import Optional
# typing 是 Python 标准库，提供类型注解支持
# 
# Optional[str] 的含义：
# - 表示这个值可以是 str 类型，也可以是 None
# - 等价于 Union[str, None]
# - 用于标注"可选"的参数或属性
#
# 类型注解的作用：
# 1. 提高代码可读性（一眼看出参数类型）
# 2. IDE 智能提示和错误检查
# 3. 文档作用
# 注意：Python 的类型注解不会在运行时强制检查，只是"提示"


# =====================================================================
# 加载环境变量
# =====================================================================

load_dotenv()
# 调用 load_dotenv() 函数，它会：
# 1. 在当前目录及父目录中查找 .env 文件
# 2. 读取文件中的 KEY=VALUE 配置
# 3. 将这些配置设置为环境变量（存入 os.environ）
# 
# 调用后，就可以用 os.getenv("KEY") 获取 .env 中的值
# 
# 为什么要在模块顶层调用？
# - 确保在使用任何配置之前，环境变量已经加载
# - 模块被导入时就会执行这行代码
#
# 如果 .env 文件不存在：
# - 不会报错，只是不加载任何配置
# - 程序会使用默认值或系统环境变量


# =====================================================================
# 配置数据类定义
# =====================================================================

@dataclass
class LLMConfig:
    """
    LLM配置数据类
    
    这个类定义了所有与 LLM（大语言模型）相关的配置项。
    使用 @dataclass 装饰器，Python 会自动生成：
    - __init__ 方法：用于创建实例
    - __repr__ 方法：用于打印对象时显示有意义的信息
    - __eq__ 方法：用于比较两个对象是否相等
    
    属性说明：
    - 每个属性都有类型注解（: str, : int 等）
    - 属性可以有默认值（= "openai"）
    - Optional[str] 表示可以是 str 或 None
    """
    
    # -----------------------------------------------------------------
    # 提供商选择
    # -----------------------------------------------------------------
    provider: str = "openai"
    # provider 属性：
    # - 类型：str（字符串）
    # - 默认值："openai"
    # - 作用：指定使用哪个 LLM 提供商
    # - 可选值："openai" 或 "anthropic"
    #
    # 语法解释：
    # provider: str = "openai"
    #    ↑       ↑      ↑
    #  属性名  类型   默认值
    
    # -----------------------------------------------------------------
    # OpenAI 配置
    # -----------------------------------------------------------------
    openai_api_key: Optional[str] = None
    # OpenAI API 密钥
    # - 类型：Optional[str]，即可以是字符串或 None
    # - 默认值：None（未设置）
    # - 作用：调用 OpenAI API 时的身份验证
    # - 获取方式：https://platform.openai.com/api-keys
    #
    # 为什么用 Optional[str] 而不是 str？
    # - 密钥可能未配置（用户可能只用 Anthropic）
    # - None 表示"未设置"，与空字符串 "" 含义不同
    
    openai_base_url: str = "https://api.openai.com/v1"
    # OpenAI API 的基础 URL
    # - 类型：str
    # - 默认值：OpenAI 官方 API 地址
    # - 作用：指定 API 请求发送到哪个服务器
    #
    # 为什么需要这个配置？
    # - 可以使用 OpenAI 兼容的第三方服务（如 Azure OpenAI）
    # - 可以使用本地代理或镜像服务
    # - 方便测试和调试
    
    openai_model: str = "gpt-4o"
    # 使用的 OpenAI 模型名称
    # - 类型：str
    # - 默认值："gpt-4o"（GPT-4 Omni，最新模型）
    # - 作用：指定调用哪个模型
    #
    # 常见模型：
    # - "gpt-4o"：最新最强，支持多模态
    # - "gpt-4-turbo"：GPT-4 快速版
    # - "gpt-3.5-turbo"：便宜快速，适合简单任务
    
    # -----------------------------------------------------------------
    # Anthropic 配置
    # -----------------------------------------------------------------
    anthropic_api_key: Optional[str] = None
    # Anthropic API 密钥
    # - 类型：Optional[str]
    # - 默认值：None
    # - 作用：调用 Claude 模型时的身份验证
    # - 获取方式：https://console.anthropic.com/
    
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    # 使用的 Anthropic 模型名称
    # - 类型：str
    # - 默认值："claude-3-5-sonnet-20241022"
    # - 作用：指定调用哪个 Claude 模型
    #
    # 常见模型：
    # - "claude-3-5-sonnet-20241022"：最新 Sonnet，性价比高
    # - "claude-3-opus-20240229"：最强大，但较贵
    # - "claude-3-haiku-20240307"：最快最便宜
    
    # -----------------------------------------------------------------
    # 通用配置
    # -----------------------------------------------------------------
    max_tokens: int = 4096
    # 最大生成 token 数
    # - 类型：int（整数）
    # - 默认值：4096
    # - 作用：限制 LLM 单次回复的最大长度
    #
    # 什么是 token？
    # - LLM 处理文本的基本单位
    # - 大约 1 个英文单词 ≈ 1 token
    # - 大约 1 个中文字 ≈ 1-2 token
    # - 4096 token ≈ 3000 个英文单词 或 2000 个中文字
    
    temperature: float = 0.7
    # 温度参数
    # - 类型：float（浮点数）
    # - 默认值：0.7
    # - 范围：0.0 ~ 2.0
    # - 作用：控制输出的随机性/创造性
    #
    # 温度的影响：
    # - 0.0：最确定性，每次输出几乎相同
    # - 0.7：平衡创造性和一致性（推荐）
    # - 1.0+：更随机，更有创意，但可能不稳定


# =====================================================================
# 配置加载函数
# =====================================================================

def load_config() -> LLMConfig:
    """
    从环境变量加载配置
    
    这个函数的作用：
    1. 读取环境变量中的配置值
    2. 如果环境变量未设置，使用默认值
    3. 创建并返回 LLMConfig 实例
    
    返回值：
    - LLMConfig 实例，包含所有配置
    
    函数签名解释：
    def load_config() -> LLMConfig:
                      ↑
                   返回类型注解
    表示这个函数返回一个 LLMConfig 类型的对象
    """
    
    return LLMConfig(
        # ---------------------------------------------------------
        # os.getenv(key, default) 函数说明：
        # - 第一个参数：环境变量的名称
        # - 第二个参数：如果环境变量不存在，返回的默认值
        # - 返回值：环境变量的值（字符串）或默认值
        #
        # 例如：os.getenv("OPENAI_API_KEY")
        # - 如果设置了 OPENAI_API_KEY=sk-xxx，返回 "sk-xxx"
        # - 如果没有设置，返回 None（因为没有提供默认值）
        # ---------------------------------------------------------
        
        provider=os.getenv("DEFAULT_LLM_PROVIDER", "openai"),
        # 从环境变量 DEFAULT_LLM_PROVIDER 读取
        # 如果未设置，默认使用 "openai"
        
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        # 从环境变量 OPENAI_API_KEY 读取
        # 如果未设置，返回 None（没有提供默认值）
        # 这是敏感信息，必须从环境变量读取，不能有默认值
        
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        # 从环境变量 OPENAI_BASE_URL 读取
        # 如果未设置，使用 OpenAI 官方 API 地址
        
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        # 从环境变量 OPENAI_MODEL 读取
        # 如果未设置，默认使用 "gpt-4o"
        
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        # 从环境变量 ANTHROPIC_API_KEY 读取
        # 如果未设置，返回 None
        
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        # 从环境变量 ANTHROPIC_MODEL 读取
        # 如果未设置，使用默认模型
        
        # 注意：max_tokens 和 temperature 没有从环境变量读取
        # 它们使用 LLMConfig 类中定义的默认值
        # 如果需要，可以添加：
        # max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
        # temperature=float(os.getenv("TEMPERATURE", "0.7")),
    )


# =====================================================================
# 全局配置实例
# =====================================================================

config = load_config()
# 创建全局配置实例
# 
# 这行代码做了什么？
# 1. 调用 load_config() 函数
# 2. 函数读取环境变量，创建 LLMConfig 实例
# 3. 将实例赋值给变量 config
#
# 为什么要创建全局实例？
# - 配置只需要加载一次
# - 其他模块可以直接导入使用：from config import config
# - 避免重复读取环境变量
#
# 使用示例（在其他文件中）：
# from config import config
# print(config.openai_api_key)  # 获取 API 密钥
# print(config.openai_model)    # 获取模型名称
#
# 这种模式叫做"单例模式"的简化版：
# - 整个程序中只有一个 config 实例
# - 所有模块共享同一个配置对象


# =====================================================================
# 补充说明：模块执行流程
# =====================================================================
"""
当其他文件执行 `from config import config` 时，会发生什么？

1. Python 首次导入 config 模块
2. 执行模块顶层代码（按顺序）：
   a. import os
   b. from dotenv import load_dotenv
   c. from dataclasses import dataclass
   d. from typing import Optional
   e. load_dotenv()  ← 加载 .env 文件
   f. 定义 LLMConfig 类（不执行，只是定义）
   g. 定义 load_config 函数（不执行，只是定义）
   h. config = load_config()  ← 执行函数，创建配置实例

3. 后续再次导入时：
   - Python 不会重新执行模块代码
   - 直接返回已创建的 config 对象
   - 这就是为什么配置只加载一次

4. 导入完成后，可以使用：
   - config.provider
   - config.openai_api_key
   - config.openai_model
   - 等等...
"""


# =====================================================================
# 补充说明：.env 文件格式
# =====================================================================
"""
.env 文件示例（创建在项目根目录）：

# OpenAI 配置
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# Anthropic 配置
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 默认提供商
DEFAULT_LLM_PROVIDER=openai

注意事项：
1. 等号两边不要有空格
2. 值不需要引号（除非包含特殊字符）
3. # 开头的行是注释
4. 文件名必须是 .env（以点开头）
"""