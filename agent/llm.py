"""
LLM API封装模块
支持OpenAI和Anthropic两种API

本模块是整个Agent系统的核心组件之一，负责：
1. 统一封装不同LLM提供商的API（OpenAI、Anthropic）
2. 处理消息格式转换（不同API的消息格式不同）
3. 支持工具调用（Function Calling / Tool Use）
4. 提供一致的调用接口，屏蔽底层差异

设计模式：策略模式（Strategy Pattern）
- LLMClient 是上下文类
- _chat_openai 和 _chat_anthropic 是两种策略
- 根据 provider 参数在运行时选择策略

==========================================================================
模块结构概览
==========================================================================

本模块包含以下主要组件：

1. Message 数据类
   - 用于表示对话中的单条消息
   - 包含角色(role)、内容(content)、工具调用(tool_calls)等属性

2. LLMClient 类
   - 统一的 LLM 客户端接口
   - 支持 OpenAI 和 Anthropic 两种后端
   - 提供 chat() 方法进行对话
   - 支持流式输出和工具调用

==========================================================================
使用示例
==========================================================================

基本使用：
```python
from agent.llm import LLMClient, Message

# 创建客户端
client = LLMClient()

# 构建消息
messages = [
    Message(role="system", content="你是一个有帮助的助手"),
    Message(role="user", content="你好，请介绍一下自己")
]

# 发送请求
response = client.chat(messages)
print(response["content"])  # 输出 AI 的回复
```

带工具调用的使用：
```python
# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }
]

# 发送带工具的请求
response = client.chat(messages, tools=tools)

# 检查是否有工具调用
if response["tool_calls"]:
    for tool_call in response["tool_calls"]:
        print(f"需要调用工具: {tool_call['function']['name']}")
```
"""

# =====================================================================
# 导入部分 (Import Section)
# =====================================================================
# 
# Python 的导入系统说明：
# 
# 1. import 语句用于导入模块或模块中的特定对象
# 2. 导入的模块会被缓存，多次导入同一模块不会重复执行
# 3. 导入顺序的惯例：
#    - 标准库（Python 自带的库）
#    - 第三方库（通过 pip 安装的库）
#    - 本地模块（项目中的其他文件）
# =====================================================================

import json
# =====================================================================
# json 模块 - Python 标准库
# =====================================================================
# 
# JSON (JavaScript Object Notation) 是一种轻量级的数据交换格式。
# 它易于人阅读和编写，也易于机器解析和生成。
# 
# json 模块提供的主要函数：
# 
# 1. json.dumps(obj) - 序列化：Python 对象 → JSON 字符串
#    示例：
#    >>> data = {"name": "Alice", "age": 25}
#    >>> json.dumps(data)
#    '{"name": "Alice", "age": 25}'
#    
#    常用参数：
#    - ensure_ascii=False: 允许非 ASCII 字符（如中文）
#    - indent=2: 格式化输出，缩进2个空格
#    
#    >>> json.dumps({"名字": "小明"}, ensure_ascii=False, indent=2)
#    '{\n  "名字": "小明"\n}'
# 
# 2. json.loads(str) - 反序列化：JSON 字符串 → Python 对象
#    示例：
#    >>> json.loads('{"name": "Alice", "age": 25}')
#    {'name': 'Alice', 'age': 25}
# 
# 3. json.dump(obj, file) - 将对象写入文件
# 4. json.load(file) - 从文件读取对象
# 
# 在本模块中的用途：
# - Anthropic API 返回的工具参数是字典，需要转换为 JSON 字符串
# - 这样可以与 OpenAI 的格式保持一致（OpenAI 返回的就是 JSON 字符串）
# =====================================================================

from typing import List, Dict, Any, Optional, Generator
# =====================================================================
# typing 模块 - Python 类型注解支持
# =====================================================================
# 
# Python 是动态类型语言，变量不需要声明类型。
# 但类型注解可以：
# 1. 提高代码可读性
# 2. 帮助 IDE 提供更好的自动补全
# 3. 配合类型检查工具（如 mypy）发现潜在错误
# 
# 注意：类型注解只是"提示"，Python 解释器不会强制检查！
# 
# 导入的类型说明：
# 
# 1. List[T] - 列表类型
#    表示元素类型为 T 的列表
#    
#    示例：
#    def process_names(names: List[str]) -> None:
#        for name in names:
#            print(name)
#    
#    List[str] 表示字符串列表，如 ["Alice", "Bob", "Charlie"]
#    List[int] 表示整数列表，如 [1, 2, 3]
#    List[Message] 表示 Message 对象的列表
# 
# 2. Dict[K, V] - 字典类型
#    表示键类型为 K、值类型为 V 的字典
#    
#    示例：
#    def get_config() -> Dict[str, Any]:
#        return {"name": "test", "count": 123}
#    
#    Dict[str, str] 表示键和值都是字符串的字典
#    Dict[str, Any] 表示键是字符串、值是任意类型的字典
# 
# 3. Any - 任意类型
#    当无法确定具体类型，或者类型可能是多种时使用
#    
#    示例：
#    def process(data: Any) -> None:
#        # data 可以是任何类型
#        pass
# 
# 4. Optional[T] - 可选类型
#    表示值可以是 T 类型或 None
#    等价于 Union[T, None]
#    
#    示例：
#    def find_user(id: int) -> Optional[str]:
#        if id == 1:
#            return "Alice"
#        return None  # 找不到时返回 None
#    
#    Optional[str] 表示可以是字符串或 None
# 
# 5. Generator[YieldType, SendType, ReturnType] - 生成器类型
#    生成器是一种特殊的迭代器，可以逐步产出值
#    
#    三个类型参数：
#    - YieldType: yield 语句产出的值的类型
#    - SendType: send() 方法接收的值的类型（通常是 None）
#    - ReturnType: return 语句返回的值的类型（通常是 None）
#    
#    示例：
#    def count_up(n: int) -> Generator[int, None, None]:
#        for i in range(n):
#            yield i  # 逐个产出 0, 1, 2, ..., n-1
#    
#    # 使用生成器
#    for num in count_up(5):
#        print(num)  # 输出 0, 1, 2, 3, 4
#    
#    生成器的优点：
#    - 惰性求值：只在需要时才计算下一个值
#    - 节省内存：不需要一次性存储所有值
#    - 适合处理大量数据或无限序列
# =====================================================================

from dataclasses import dataclass
# =====================================================================
# dataclass 装饰器 - 简化数据类定义
# =====================================================================
# 
# 什么是装饰器（Decorator）？
# 装饰器是一种特殊的函数，用于修改其他函数或类的行为。
# 使用 @ 符号放在函数或类定义之前。
# 
# @dataclass 装饰器的作用：
# 自动为类生成常用的特殊方法，减少样板代码。
# 
# 自动生成的方法：
# 
# 1. __init__() - 构造方法
#    根据类属性自动生成初始化代码
#    
# 2. __repr__() - 字符串表示
#    返回对象的可读字符串，用于调试
#    
# 3. __eq__() - 相等比较
#    比较两个对象的所有属性是否相等
# 
# 对比示例：
# 
# 不使用 @dataclass（需要手写很多代码）：
# ```python
# class Person:
#     def __init__(self, name: str, age: int):
#         self.name = name
#         self.age = age
#     
#     def __repr__(self):
#         return f"Person(name={self.name!r}, age={self.age!r})"
#     
#     def __eq__(self, other):
#         if not isinstance(other, Person):
#             return False
#         return self.name == other.name and self.age == other.age
# ```
# 
# 使用 @dataclass（简洁很多）：
# ```python
# @dataclass
# class Person:
#     name: str
#     age: int
# ```
# 
# 两种写法的效果完全相同！
# 
# @dataclass 的常用参数：
# - frozen=True: 创建不可变对象（属性不能修改）
# - order=True: 自动生成比较方法（<, <=, >, >=）
# - slots=True: 使用 __slots__ 优化内存（Python 3.10+）
# =====================================================================

from config import config
# =====================================================================
# 从 config 模块导入配置实例
# =====================================================================
# 
# 这是相对导入，从项目根目录的 config.py 文件导入 config 对象。
# 
# config 是 Config 类的实例，包含所有配置信息：
# - config.provider: LLM 提供商（"openai" 或 "anthropic"）
# - config.openai_api_key: OpenAI API 密钥
# - config.openai_base_url: OpenAI API 基础 URL
# - config.openai_model: OpenAI 模型名称
# - config.anthropic_api_key: Anthropic API 密钥
# - config.anthropic_model: Anthropic 模型名称
# - config.max_tokens: 最大生成 token 数
# - config.temperature: 温度参数（控制随机性）
# 
# 导入语法说明：
# - `from module import name`: 从模块导入特定名称
# - `import module`: 导入整个模块
# 
# 区别：
# ```python
# from config import config
# print(config.provider)  # 直接使用 config
# 
# import config
# print(config.config.provider)  # 需要通过模块名访问
# ```
# =====================================================================


# =====================================================================
# 消息数据类 (Message Data Class)
# =====================================================================

@dataclass
class Message:
    """
    消息数据类 - LLM对话的基本单位
    
    =================================================================
    概述
    =================================================================
    
    在与 LLM（大语言模型）对话时，所有的交互都以"消息"为单位。
    每条消息包含两个核心要素：
    1. 角色（role）：谁发送的这条消息
    2. 内容（content）：消息的具体内容
    
    这个类用于在程序内部统一表示消息，然后根据不同的 API 
    转换为对应的格式（OpenAI 和 Anthropic 的消息格式略有不同）。
    
    =================================================================
    消息角色详解
    =================================================================
    
    1. system（系统消息）
       - 用于设定 AI 的行为、人设、规则
       - 通常放在对话开头
       - AI 会遵循系统消息中的指示
       - 示例："你是一个专业的Python程序员，请用简洁的语言回答问题"
    
    2. user（用户消息）
       - 用户输入的内容
       - 可以是问题、指令、或任何文本
       - 示例："请帮我写一个快速排序算法"
    
    3. assistant（助手消息）
       - AI 的回复
       - 在多轮对话中，需要包含之前的 AI 回复
       - 示例："好的，这是快速排序的实现：..."
    
    4. tool（工具消息）
       - 工具执行的结果
       - 当 AI 调用工具后，需要将结果以 tool 消息返回
       - 示例：文件内容、命令执行结果等
    
    =================================================================
    对话流程示例
    =================================================================
    
    一个典型的带工具调用的对话流程：
    
    1. [system] "你是一个文件助手，可以读取和写入文件"
    2. [user] "请读取 config.py 文件"
    3. [assistant] (AI 决定调用 read_file 工具)
       - content: "" (可能为空)
       - tool_calls: [{"name": "read_file", "arguments": {"path": "config.py"}}]
    4. [tool] "文件内容是：..." (工具执行结果)
       - tool_call_id: "call_xxx" (关联到步骤3的调用)
    5. [assistant] "config.py 文件的内容如下：..."
    
    =================================================================
    使用示例
    =================================================================
    
    ```python
    # 创建系统消息
    system_msg = Message(
        role="system",
        content="你是一个有帮助的助手"
    )
    
    # 创建用户消息
    user_msg = Message(
        role="user",
        content="你好"
    )
    
    # 创建带工具调用的助手消息
    assistant_msg = Message(
        role="assistant",
        content="",
        tool_calls=[{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "test.txt"}'
            }
        }]
    )
    
    # 创建工具结果消息
    tool_msg = Message(
        role="tool",
        content="文件内容...",
        tool_call_id="call_123"
    )
    ```
    """
    
    # =================================================================
    # 类属性定义
    # =================================================================
    # 
    # 在 @dataclass 中，类属性的定义方式：
    # 属性名: 类型 = 默认值
    # 
    # 如果没有默认值，该属性是必需的（创建对象时必须提供）
    # 如果有默认值，该属性是可选的
    # 
    # 注意：有默认值的属性必须放在没有默认值的属性后面！
    # 否则会报错：non-default argument follows default argument
    # =================================================================
    
    role: str
    # =================================================================
    # role 属性 - 消息角色
    # =================================================================
    # 
    # 类型：str（字符串）
    # 必需：是（没有默认值）
    # 
    # 可选值：
    # - "system": 系统消息，设定 AI 的行为
    # - "user": 用户消息，用户的输入
    # - "assistant": 助手消息，AI 的回复
    # - "tool": 工具消息，工具执行的结果
    # 
    # 为什么使用字符串而不是枚举？
    # 1. 与 API 格式保持一致（API 使用字符串）
    # 2. 更灵活，便于扩展
    # 3. 序列化/反序列化更简单
    # =================================================================
    
    content: str
    # =================================================================
    # content 属性 - 消息内容
    # =================================================================
    # 
    # 类型：str（字符串）
    # 必需：是（没有默认值）
    # 
    # 存储消息的实际文本内容。
    # 
    # 不同角色的 content 含义：
    # - system: AI 的行为指示
    # - user: 用户的问题或指令
    # - assistant: AI 的回复文本
    # - tool: 工具执行的结果
    # 
    # 注意：当 assistant 消息包含 tool_calls 时，
    # content 可能为空字符串（AI 只是调用工具，没有文本回复）
    # =================================================================
    
    tool_calls: Optional[List[Dict]] = None
    # =================================================================
    # tool_calls 属性 - 工具调用列表
    # =================================================================
    # 
    # 类型：Optional[List[Dict]]
    #       即 List[Dict] 或 None
    # 必需：否（默认值为 None）
    # 
    # 当 AI 决定调用工具时，这个字段会包含工具调用信息。
    # 只有 role="assistant" 的消息才可能有这个字段。
    # 
    # 数据结构示例：
    # [
    #     {
    #         "id": "call_abc123",           # 调用ID，唯一标识
    #         "type": "function",            # 类型，固定为 "function"
    #         "function": {
    #             "name": "read_file",       # 工具/函数名称
    #             "arguments": '{"path": "config.py"}'  # 参数（JSON字符串）
    #         }
    #     },
    #     {
    #         "id": "call_def456",           # 可以同时调用多个工具
    #         "type": "function",
    #         "function": {
    #             "name": "list_directory",
    #             "arguments": '{"path": "."}'
    #         }
    #     }
    # ]
    # 
    # 为什么 arguments 是 JSON 字符串而不是字典？
    # 1. 这是 OpenAI API 的格式
    # 2. 保持原始格式，避免解析错误
    # 3. 在需要时再解析：json.loads(arguments)
    # 
    # 默认值 None 表示这条消息没有工具调用
    # =================================================================
    
    tool_call_id: Optional[str] = None
    # =================================================================
    # tool_call_id 属性 - 工具调用ID
    # =================================================================
    # 
    # 类型：Optional[str]
    #       即 str 或 None
    # 必需：否（默认值为 None）
    # 
    # 当 role="tool" 时，这个字段用于关联到原始的工具调用。
    # 
    # 为什么需要这个ID？
    # ---------------------------------------------------------
    # 
    # 场景：AI 可能同时调用多个工具
    # 
    # 例如，AI 说："我需要读取两个文件"
    # tool_calls = [
    #     {"id": "call_1", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
    #     {"id": "call_2", "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'}}
    # ]
    # 
    # 执行后，需要返回两个结果：
    # Message(role="tool", content="a.txt的内容", tool_call_id="call_1")
    # Message(role="tool", content="b.txt的内容", tool_call_id="call_2")
    # 
    # 通过 tool_call_id，AI 可以知道哪个结果对应哪个调用。
    # 
    # 如果没有这个ID，当两个工具调用的结果都是文件内容时，
    # AI 就无法区分哪个是 a.txt 的内容，哪个是 b.txt 的内容。
    # =================================================================


# =====================================================================
# LLM客户端类 (LLM Client Class)
# =====================================================================

class LLMClient:
    """
    统一的LLM客户端 - 封装不同LLM提供商的API
    
    =================================================================
    设计目标
    =================================================================
    
    这个类的设计目标是提供一个统一的接口，让调用者不需要关心
    底层使用的是 OpenAI 还是 Anthropic。
    
    好处：
    1. 调用代码简洁：只需要调用 chat() 方法
    2. 易于切换：只需修改配置，不需要修改调用代码
    3. 易于扩展：添加新的提供商只需要添加新的方法
    
    =================================================================
    设计模式：策略模式（Strategy Pattern）
    =================================================================
    
    策略模式是一种行为设计模式，它定义了一系列算法，
    将每个算法封装起来，并使它们可以互换。
    
    在本类中：
    - chat() 方法是统一接口（上下文）
    - _chat_openai() 是策略1
    - _chat_anthropic() 是策略2
    - self.provider 决定使用哪种策略
    
    类比：
    想象你要从北京到上海，可以选择：
    - 飞机（快但贵）
    - 高铁（适中）
    - 汽车（慢但灵活）
    
    无论选择哪种方式，目标都是"到达上海"。
    策略模式就是把这些不同的"方式"封装起来，
    让调用者只需要说"我要去上海"，而不用关心具体怎么去。
    
    =================================================================
    类的结构
    =================================================================
    
    属性：
    - provider: str - LLM提供商名称
    - _client: Any - 实际的API客户端对象
    
    公开方法：
    - __init__(): 构造方法，初始化客户端
    - chat(): 发送聊天请求，返回响应
    
    私有方法（以下划线开头）：
    - _init_client(): 初始化API客户端
    - _chat_openai(): OpenAI API调用实现
    - _chat_anthropic(): Anthropic API调用实现
    - _stream_openai(): OpenAI流式输出实现
    - _convert_tools_to_anthropic(): 工具格式转换
    
    =================================================================
    使用示例
    =================================================================
    
    基本使用：
    ```python
    # 创建客户端（使用配置文件中的默认提供商）
    client = LLMClient()
    
    # 或者指定提供商
    client = LLMClient(provider="anthropic")
    
    # 构建消息列表
    messages = [
        Message(role="system", content="你是一个助手"),
        Message(role="user", content="你好")
    ]
    
    # 发送请求
    response = client.chat(messages)
    
    # 处理响应
    print(response["content"])  # AI 的回复文本
    print(response["tool_calls"])  # 工具调用（如果有）
    print(response["finish_reason"])  # 结束原因
    ```
    
    带工具的使用：
    ```python
    # 定义工具（OpenAI格式）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    # 发送带工具的请求
    response = client.chat(messages, tools=tools)
    
    # 检查是否需要调用工具
    if response["tool_calls"]:
        for tool_call in response["tool_calls"]:
            name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            print(f"需要调用: {name}({args})")
    ```
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        初始化LLM客户端
        
        =============================================================
        方法签名解析
        =============================================================
        
        def __init__(self, provider: Optional[str] = None):
            ↑    ↑      ↑        ↑                    ↑
            │    │      │        │                    └─ 默认值
            │    │      │        └─ 类型注解：可选字符串
            │    │      └─ 参数名
            │    └─ 方法名（双下划线开头和结尾是特殊方法）
            └─ 函数定义关键字
        
        __init__ 是 Python 的特殊方法（也叫魔术方法/dunder方法）：
        - 在创建对象时自动调用
        - 用于初始化对象的属性
        """
        
        self.provider = provider or config.provider
        # 设置提供商
        # 
        # `provider or config.provider` 的含义：
        # - 如果 provider 有值（不是 None 或空字符串），使用 provider
        # - 否则使用 config.provider（配置文件中的默认值）
        # 
        # 这是 Python 的短路求值（short-circuit evaluation）：
        # - `a or b`：如果 a 为真，返回 a；否则返回 b
        # - None、空字符串、0、空列表等都被视为"假"
        
        self._client = None
        # 存储实际的 API 客户端对象
        # 
        # 命名约定：以下划线开头的属性（_client）表示"私有"
        # Python 没有真正的私有属性，这只是一种约定：
        # - 告诉其他开发者"这是内部使用的，不要直接访问"
        # - IDE 通常不会在自动补全中显示这些属性
        
        self._init_client()
        # 调用初始化方法，创建实际的 API 客户端
        # 这是一种常见的模式：在 __init__ 中调用其他方法来组织代码
    
    def _init_client(self):
        """
        初始化对应的API客户端
        
        根据 self.provider 的值，创建对应的 API 客户端对象。
        这里使用了延迟导入（lazy import）：
        - 只在需要时才导入对应的库
        - 如果用户只用 OpenAI，就不需要安装 anthropic 库
        """
        
        if self.provider == "openai":
            # ---------------------------------------------------------
            # OpenAI 客户端初始化
            # ---------------------------------------------------------
            
            from openai import OpenAI
            # 延迟导入 OpenAI 类
            # 
            # 为什么在这里导入而不是在文件开头？
            # 1. 延迟加载：只有真正使用时才导入，加快启动速度
            # 2. 可选依赖：如果用户只用 Anthropic，不需要安装 openai 库
            # 3. 避免导入错误：如果库未安装，只有在使用时才会报错
            
            self._client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url
            )
            # 创建 OpenAI 客户端实例
            # 
            # 参数说明：
            # - api_key: API 密钥，用于身份验证
            # - base_url: API 基础 URL，可以指向官方或第三方服务
            #
            # 这个客户端对象提供了调用 OpenAI API 的方法
            
        elif self.provider == "anthropic":
            # ---------------------------------------------------------
            # Anthropic 客户端初始化
            # ---------------------------------------------------------
            
            from anthropic import Anthropic
            # 延迟导入 Anthropic 类
            
            self._client = Anthropic(
                api_key=config.anthropic_api_key
            )
            # 创建 Anthropic 客户端实例
            # Anthropic 不需要 base_url，因为只有官方服务
            
        else:
            # ---------------------------------------------------------
            # 不支持的提供商
            # ---------------------------------------------------------
            
            raise ValueError(f"不支持的LLM提供商: {self.provider}")
            # raise 语句用于抛出异常
            # 
            # ValueError 是 Python 内置异常，表示"值错误"
            # 当参数的值不在预期范围内时使用
            #
            # f"..." 是 f-string（格式化字符串）：
            # - {self.provider} 会被替换为实际的值
            # - 例如：如果 provider="google"，消息是 "不支持的LLM提供商: google"
            #
            # 抛出异常后，程序会停止执行（除非被 try-except 捕获）
    
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        发送聊天请求 - 这是主要的公开接口
        
        这个方法是 LLMClient 的核心，提供统一的调用方式。
        无论底层使用 OpenAI 还是 Anthropic，调用方式都相同。
        
        参数说明：
        
        messages: List[Message]
            消息列表，包含对话历史
            示例：
            [
                Message(role="system", content="你是助手"),
                Message(role="user", content="你好"),
                Message(role="assistant", content="你好！有什么可以帮你？"),
                Message(role="user", content="读取config.py")
            ]
        
        tools: Optional[List[Dict]] = None
            可用工具列表，OpenAI Function Calling 格式
            示例：
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "读取文件内容",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "文件路径"}
                            },
                            "required": ["path"]
                        }
                    }
                }
            ]
        
        stream: bool = False
            是否使用流式输出
            - False: 等待完整响应后返回（默认）
            - True: 逐步返回响应片段（打字机效果）
        
        返回值：Dict[str, Any]
            包含以下字段的字典：
            {
                "content": "AI的回复文本",
                "tool_calls": [...] 或 None,  # 工具调用列表
                "finish_reason": "stop" 或 "tool_calls"  # 结束原因
            }
        
        方法签名解释：
        def chat(self, messages: List[Message], ...) -> Dict[str, Any]:
                                                      ↑
                                               返回类型注解
        """
        
        if self.provider == "openai":
            return self._chat_openai(messages, tools, stream)
        else:
            return self._chat_anthropic(messages, tools, stream)
        # 根据提供商选择对应的实现方法
        # 这就是策略模式的核心：运行时选择算法
    
    def _chat_openai(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        OpenAI API调用实现
        
        这个方法负责：
        1. 将内部 Message 格式转换为 OpenAI API 格式
        2. 构建请求参数
        3. 发送请求并处理响应
        4. 将响应转换为统一格式返回
        """
        
        # =============================================================
        # 设置提供商
        # =============================================================
        # OpenAI API 需要的消息格式是字典列表，而不是 Message 对象
        # 需要将 Message 对象转换为字典
        
        formatted_messages = []
        # 创建空列表，用于存储转换后的消息
        
        for msg in messages:
            # 遍历消息列表
            # for...in 循环：依次取出列表中的每个元素
            
            formatted_msg = {"role": msg.role, "content": msg.content}
            # 创建基本的消息字典
            # 
            # 字典语法：{键: 值, 键: 值, ...}
            # msg.role 和 msg.content 是访问 Message 对象的属性
            
            if msg.tool_calls:
                formatted_msg["tool_calls"] = msg.tool_calls
            # 如果消息包含工具调用，添加到字典中
            # 
            # if 条件判断：
            # - msg.tool_calls 如果是 None 或空列表，条件为 False
            # - 如果有内容，条件为 True
            
            if msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            # 如果是工具结果消息，添加关联 ID
            
            formatted_messages.append(formatted_msg)
            # 将转换后的消息添加到列表
            # list.append(item) 在列表末尾添加元素
        
        # =============================================================
        # 第二步：构建请求参数
        # =============================================================
        
        kwargs = {
            "model": config.openai_model,
            "messages": formatted_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        # kwargs 是 "keyword arguments" 的缩写
        # 这是一个字典，包含要传递给 API 的所有参数
        #
        # 参数说明：
        # - model: 使用的模型名称，如 "gpt-4o"
        # - messages: 对话历史
        # - max_tokens: 最大生成长度
        # - temperature: 随机性控制
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        # 如果提供了工具列表，添加到请求参数
        # 
        # tool_choice 控制工具调用行为：
        # - "auto": 让模型自动决定是否调用工具
        # - "none": 禁止调用工具
        # - {"type": "function", "function": {"name": "xxx"}}: 强制调用指定工具
        
        if stream:
            return self._stream_openai(kwargs)
        # 如果需要流式输出，调用流式方法
        # 流式输出会返回一个生成器，逐步产出响应片段
        
        # =============================================================
        # 第三步：发送请求
        # =============================================================
        
        response = self._client.chat.completions.create(**kwargs)
        # 调用 OpenAI API
        #
        # **kwargs 是字典解包语法：
        # 将字典中的键值对作为关键字参数传递
        # 等价于：
        # self._client.chat.completions.create(
        #     model=config.openai_model,
        #     messages=formatted_messages,
        #     max_tokens=config.max_tokens,
        #     temperature=config.temperature,
        #     tools=tools,  # 如果有的话
        #     tool_choice="auto"  # 如果有的话
        # )
        #
        # response 是 OpenAI 返回的响应对象
        
        # =============================================================
        # 第四步：处理响应
        # =============================================================
        
        result = {
            "content": response.choices[0].message.content or "",
            "tool_calls": None,
            "finish_reason": response.choices[0].finish_reason
        }
        # 构建统一格式的返回结果
        #
        # response.choices[0] 是什么？
        # - OpenAI API 可以一次生成多个回复（通过 n 参数）
        # - choices 是一个列表，包含所有生成的回复
        # - [0] 取第一个（通常只有一个）
        #
        # response.choices[0].message.content
        # - message 是回复消息对象
        # - content 是文本内容
        # - `or ""` 处理 content 为 None 的情况（工具调用时可能为空）
        #
        # finish_reason 表示生成结束的原因：
        # - "stop": 正常结束
        # - "tool_calls": 需要调用工具
        # - "length": 达到最大长度限制
        
        # 处理工具调用
        if response.choices[0].message.tool_calls:
            # 如果 AI 决定调用工具
            
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
            # 这是列表推导式（list comprehension）
            # 
            # 语法：[表达式 for 变量 in 可迭代对象]
            # 
            # 等价于：
            # result["tool_calls"] = []
            # for tc in response.choices[0].message.tool_calls:
            #     result["tool_calls"].append({
            #         "id": tc.id,
            #         "type": "function",
            #         "function": {
            #             "name": tc.function.name,
            #             "arguments": tc.function.arguments
            #         }
            #     })
            #
            # tc 是 tool_call 的缩写，代表每个工具调用对象
            # tc.id: 调用ID
            # tc.function.name: 工具名称
            # tc.function.arguments: 参数（JSON字符串）
        
        return result
        # 返回处理后的结果字典
    
    def _stream_openai(self, kwargs: Dict) -> Generator[Dict, None, None]:
        """
        OpenAI流式输出实现
        
        =============================================================
        什么是流式输出？
        =============================================================
        
        流式输出（Streaming）是一种数据传输方式：
        - 非流式：等待所有数据生成完毕后，一次性返回
        - 流式：边生成边返回，逐步传输数据
        
        类比：
        - 非流式像是等整部电影下载完再看
        - 流式像是边下载边播放（在线观看）
        
        流式输出的优点：
        1. 用户体验好：可以看到"打字机效果"，感觉响应更快
        2. 首字节时间短：不需要等待完整响应
        3. 可以提前中断：如果发现回答不对，可以提前停止
        
        =============================================================
        生成器（Generator）详解
        =============================================================
        
        这个方法返回一个生成器，使用 yield 关键字产出值。
        
        生成器 vs 普通函数：
        - 普通函数：使用 return 返回值，函数执行结束
        - 生成器：使用 yield 产出值，函数暂停，下次调用继续执行
        
        示例：
        ```python
        # 普通函数 - 一次性返回所有数据
        def get_numbers():
            return [1, 2, 3, 4, 5]  # 需要先生成完整列表
        
        # 生成器 - 逐个产出数据
        def generate_numbers():
            for i in range(1, 6):
                yield i  # 每次产出一个，暂停等待下次调用
        
        # 使用生成器
        for num in generate_numbers():
            print(num)  # 依次打印 1, 2, 3, 4, 5
        ```
        
        生成器的内存优势：
        - 普通函数：[1, 2, ..., 1000000] 需要存储100万个数
        - 生成器：每次只产出一个数，内存占用极小
        
        =============================================================
        参数说明
        =============================================================
        
        kwargs: Dict
            请求参数字典，包含 model、messages、max_tokens 等
            这个方法会修改 kwargs，添加 stream=True
        
        返回值: Generator[Dict, None, None]
            生成器，每次产出一个字典：
            - {"type": "content", "content": "文本片段"}
            - {"type": "tool_call", "tool_calls": [...]}
        
        =============================================================
        使用示例
        =============================================================
        
        ```python
        # 流式输出的使用方式
        for chunk in client._stream_openai(kwargs):
            if chunk["type"] == "content":
                print(chunk["content"], end="", flush=True)  # 实时打印
        ```
        """
        
        kwargs["stream"] = True
        # 在请求参数中启用流式模式
        # 这会让 OpenAI API 返回一个可迭代的流对象，而不是完整响应
        
        stream = self._client.chat.completions.create(**kwargs)
        # 发送请求，获取流对象
        # 
        # 注意：这里的 stream 不是完整的响应，而是一个可迭代对象
        # 每次迭代会获取一个"块"（chunk），包含部分响应内容
        
        for chunk in stream:
            # 遍历流中的每个块
            # 
            # chunk 的结构类似于普通响应，但内容是增量的：
            # - chunk.choices[0].delta.content: 新增的文本内容
            # - chunk.choices[0].delta.tool_calls: 新增的工具调用信息
            # 
            # delta（增量）vs message（完整）：
            # - 非流式响应使用 message，包含完整内容
            # - 流式响应使用 delta，只包含新增部分
            
            if chunk.choices[0].delta.content:
                yield {"type": "content", "content": chunk.choices[0].delta.content}
                # yield 关键字：产出一个值，暂停函数执行
                # 
                # 当调用者请求下一个值时，函数从这里继续执行
                # 
                # 产出的字典格式：
                # {
                #     "type": "content",  # 类型标识，表示这是文本内容
                #     "content": "你"     # 实际的文本片段（可能只有一个字）
                # }
                
            if chunk.choices[0].delta.tool_calls:
                yield {"type": "tool_call", "tool_calls": chunk.choices[0].delta.tool_calls}
                # 如果有工具调用信息，也逐步产出
                # 
                # 工具调用的流式输出比较复杂：
                # - 工具名称可能分多次传输
                # - 参数（JSON字符串）也可能分多次传输
                # - 调用者需要自己拼接这些片段
    
    def _chat_anthropic(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Anthropic API调用实现
        
        =============================================================
        Anthropic API vs OpenAI API 的主要区别
        =============================================================
        
        1. System 消息处理方式不同：
           - OpenAI: system 消息放在 messages 列表中
           - Anthropic: system 消息作为单独的参数传递
        
        2. 工具调用格式不同：
           - OpenAI: 使用 "function" 格式
           - Anthropic: 使用 "tool_use" 和 "tool_result" 格式
        
        3. 响应结构不同：
           - OpenAI: response.choices[0].message.content
           - Anthropic: response.content（是一个块列表）
        
        4. 工具结果返回方式不同：
           - OpenAI: role="tool" 的消息
           - Anthropic: role="user" 的消息，content 中包含 tool_result
        
        =============================================================
        方法流程
        =============================================================
        
        1. 提取 system 消息（Anthropic 需要单独处理）
        2. 转换消息格式（特别是 tool 消息）
        3. 构建请求参数
        4. 发送请求
        5. 处理响应，转换为统一格式
        """
        
        # =============================================================
        # 第一步：提取 system 消息并转换消息格式
        # =============================================================
        
        system_content = ""
        # 存储 system 消息的内容
        # Anthropic API 要求 system 消息单独传递，不能放在 messages 中
        
        formatted_messages = []
        # 存储转换后的消息列表
        
        for msg in messages:
            # 遍历所有消息，根据角色进行不同处理
            
            if msg.role == "system":
                # ---------------------------------------------------------
                # 处理 system 消息
                # ---------------------------------------------------------
                # Anthropic 的 system 消息不放在 messages 中
                # 而是作为 API 的单独参数
                system_content = msg.content
                # 注意：这里没有 append 到 formatted_messages
                # system 消息会在后面单独处理
                
            elif msg.role == "tool":
                # ---------------------------------------------------------
                # 处理 tool 消息（工具执行结果）
                # ---------------------------------------------------------
                # 
                # Anthropic 的工具结果格式与 OpenAI 完全不同！
                # 
                # OpenAI 格式：
                # {
                #     "role": "tool",
                #     "content": "工具执行结果",
                #     "tool_call_id": "call_xxx"
                # }
                # 
                # Anthropic 格式：
                # {
                #     "role": "user",  # 注意：角色是 user，不是 tool！
                #     "content": [
                #         {
                #             "type": "tool_result",
                #             "tool_use_id": "call_xxx",
                #             "content": "工具执行结果"
                #         }
                #     ]
                # }
                # 
                # 为什么 Anthropic 用 "user" 角色？
                # - Anthropic 的设计理念：工具结果是"用户提供的信息"
                # - 这样可以保持 user/assistant 交替的对话结构
                
                formatted_messages.append({
                    "role": "user",  # Anthropic 要求使用 "user" 角色
                    "content": [
                        {
                            "type": "tool_result",  # 标识这是工具结果
                            "tool_use_id": msg.tool_call_id,  # 关联到原始调用
                            "content": msg.content  # 工具执行的结果
                        }
                    ]
                })
                # 注意 content 是一个列表，不是字符串！
                # Anthropic 的 content 可以包含多种类型的块（block）
                
            else:
                # ---------------------------------------------------------
                # 处理其他消息（user 和 assistant）
                # ---------------------------------------------------------
                # user 和 assistant 消息的格式相对简单
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # =============================================================
        # 第二步：构建请求参数
        # =============================================================
        
        kwargs = {
            "model": config.anthropic_model,
            # 使用配置中的 Anthropic 模型名称
            # 例如："claude-3-5-sonnet-20241022"
            
            "max_tokens": config.max_tokens,
            # 最大生成 token 数
            # Anthropic 要求必须指定这个参数（OpenAI 是可选的）
            
            "messages": formatted_messages,
            # 转换后的消息列表
        }
        
        if system_content:
            kwargs["system"] = system_content
            # 如果有 system 消息，作为单独参数传递
            # 
            # 这是 Anthropic API 的特殊要求：
            # - system 不能放在 messages 中
            # - 必须作为顶级参数传递
        
        if tools:
            kwargs["tools"] = self._convert_tools_to_anthropic(tools)
            # 如果有工具，需要转换格式
            # OpenAI 和 Anthropic 的工具定义格式不同
            # 详见 _convert_tools_to_anthropic 方法
        
        # =============================================================
        # 第三步：发送请求
        # =============================================================
        
        response = self._client.messages.create(**kwargs)
        # 调用 Anthropic API
        # 
        # 注意 API 路径的区别：
        # - OpenAI: client.chat.completions.create()
        # - Anthropic: client.messages.create()
        
        # =============================================================
        # 第四步：处理响应
        # =============================================================
        
        result = {
            "content": "",
            # 初始化为空字符串，后面会拼接
            
            "tool_calls": None,
            # 初始化为 None，如果有工具调用会变成列表
            
            "finish_reason": response.stop_reason
            # Anthropic 使用 stop_reason 而不是 finish_reason
            # 可能的值：
            # - "end_turn": 正常结束（对应 OpenAI 的 "stop"）
            # - "tool_use": 需要调用工具（对应 OpenAI 的 "tool_calls"）
            # - "max_tokens": 达到最大长度
        }
        
        # ---------------------------------------------------------
        # 处理响应内容
        # ---------------------------------------------------------
        # 
        # Anthropic 的响应结构：
        # response.content 是一个"块"（block）列表
        # 每个块可以是不同类型：
        # - text: 文本内容
        # - tool_use: 工具调用
        # 
        # 示例：
        # response.content = [
        #     TextBlock(type="text", text="让我来读取文件"),
        #     ToolUseBlock(type="tool_use", id="xxx", name="read_file", input={...})
        # ]
        
        for block in response.content:
            # 遍历响应中的每个块
            
            if block.type == "text":
                # ---------------------------------------------------------
                # 处理文本块
                # ---------------------------------------------------------
                result["content"] += block.text
                # 将文本内容拼接到结果中
                # 
                # 为什么用 += 而不是直接赋值？
                # 因为可能有多个文本块，需要拼接
                
            elif block.type == "tool_use":
                # ---------------------------------------------------------
                # 处理工具调用块
                # ---------------------------------------------------------
                # 
                # Anthropic 的工具调用格式：
                # ToolUseBlock(
                #     type="tool_use",
                #     id="toolu_xxx",        # 调用ID
                #     name="read_file",      # 工具名称
                #     input={"path": "..."}  # 参数（字典，不是JSON字符串！）
                # )
                # 
                # 需要转换为统一格式（OpenAI 格式）
                
                if result["tool_calls"] is None:
                    result["tool_calls"] = []
                    # 第一次遇到工具调用时，初始化列表
                    # 
                    # 为什么不在开头就初始化为 []？
                    # 因为如果没有工具调用，返回 None 更清晰
                    # 调用者可以用 if result["tool_calls"]: 来判断
                
                result["tool_calls"].append({
                    "id": block.id,
                    # 工具调用ID，用于关联结果
                    
                    "type": "function",
                    # 统一使用 OpenAI 的格式
                    
                    "function": {
                        "name": block.name,
                        # 工具名称
                        
                        "arguments": json.dumps(block.input)
                        # 关键转换！
                        # Anthropic 返回的是字典（block.input）
                        # OpenAI 格式要求是 JSON 字符串
                        # 所以用 json.dumps() 转换
                        # 
                        # 例如：
                        # block.input = {"path": "config.py"}
                        # json.dumps(block.input) = '{"path": "config.py"}'
                    }
                })
        
        return result
        # 返回统一格式的结果
        # 调用者不需要知道底层用的是 OpenAI 还是 Anthropic
    
    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """
        将OpenAI格式的工具定义转换为Anthropic格式
        
        =============================================================
        为什么需要格式转换？
        =============================================================
        
        OpenAI 和 Anthropic 的工具定义格式不同：
        
        OpenAI 格式：
        ```json
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取文件内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径"}
                    },
                    "required": ["path"]
                }
            }
        }
        ```
        
        Anthropic 格式：
        ```json
        {
            "name": "read_file",
            "description": "读取文件内容",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        }
        ```
        
        主要区别：
        1. OpenAI 有外层的 "type": "function" 和 "function" 包装
        2. Anthropic 直接是扁平结构
        3. OpenAI 用 "parameters"，Anthropic 用 "input_schema"
        
        =============================================================
        设计决策
        =============================================================
        
        本项目选择使用 OpenAI 格式作为内部标准格式，原因：
        1. OpenAI 是更早、更广泛使用的格式
        2. 很多工具和库都支持 OpenAI 格式
        3. 只需要在调用 Anthropic 时转换一次
        
        =============================================================
        参数说明
        =============================================================
        
        tools: List[Dict]
            OpenAI 格式的工具定义列表
        
        返回值: List[Dict]
            Anthropic 格式的工具定义列表
        """
        
        anthropic_tools = []
        # 存储转换后的工具列表
        
        for tool in tools:
            # 遍历每个工具定义
            
            if tool["type"] == "function":
                # 只处理 function 类型的工具
                # （目前 OpenAI 只支持 function 类型，但保留这个检查以防未来扩展）
                
                func = tool["function"]
                # 获取 function 对象，包含 name、description、parameters
                
                anthropic_tools.append({
                    "name": func["name"],
                    # 工具名称，直接复制
                    
                    "description": func.get("description", ""),
                    # 工具描述
                    # 
                    # dict.get(key, default) 方法：
                    # - 如果 key 存在，返回对应的值
                    # - 如果 key 不存在，返回 default（这里是空字符串）
                    # 
                    # 为什么用 get 而不是 func["description"]？
                    # 因为 description 可能不存在，直接访问会抛出 KeyError
                    # get 方法更安全
                    
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                    # 参数模式（JSON Schema 格式）
                    # 
                    # OpenAI 叫 "parameters"
                    # Anthropic 叫 "input_schema"
                    # 内容格式是一样的（都是 JSON Schema）
                    # 
                    # 默认值 {"type": "object", "properties": {}} 表示：
                    # - 参数是一个对象
                    # - 没有任何属性（即不需要参数）
                })
        
        return anthropic_tools
        # 返回转换后的工具列表
