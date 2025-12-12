"""
工具系统模块 (Tool System Module)
================================

本模块定义了工具的注册、管理和执行机制。
这是一个典型的"插件系统"或"工具注册表"设计模式的实现。

主要功能：
1. 定义工具的数据结构（Tool 类）
2. 提供工具注册表（ToolRegistry 类）用于管理所有工具
3. 实现装饰器模式，方便注册新工具
4. 提供内置的常用工具（文件操作、命令执行等）

设计模式：
- 注册表模式 (Registry Pattern): 集中管理所有工具
- 装饰器模式 (Decorator Pattern): 使用 @registry.register() 注册工具
- 数据类模式 (Dataclass Pattern): 使用 @dataclass 简化类定义

这个模块是 Agent 系统的核心组件之一，LLM 可以通过调用这些工具
来执行实际的操作（如读写文件、执行命令等）。
"""

# ============================================================================
# 导入语句 (Import Statements)
# ============================================================================

import json
# json 模块：用于处理 JSON 格式的数据
# 虽然在当前代码中没有直接使用，但可能在工具执行时需要

from typing import Dict, Any, Callable, List, Optional
# typing 模块：Python 的类型提示系统
# - Dict[K, V]: 字典类型，K 是键类型，V 是值类型
#   例如 Dict[str, Any] 表示键为字符串、值为任意类型的字典
# - Any: 表示任意类型，当无法确定具体类型时使用
# - Callable: 表示可调用对象（函数、方法、lambda 等）
#   Callable[[参数类型], 返回类型] 或简单的 Callable
# - List[T]: 列表类型，T 是元素类型
# - Optional[T]: 可选类型，等价于 Union[T, None]，表示值可以是 T 类型或 None

from dataclasses import dataclass, field
# dataclasses 模块：Python 3.7+ 引入的数据类装饰器
# - @dataclass: 自动生成 __init__, __repr__, __eq__ 等方法
# - field(): 用于自定义字段的默认值和其他属性


# ============================================================================
# Tool 数据类定义 (Tool Dataclass Definition)
# ============================================================================

@dataclass
class Tool:
    """
    工具定义类 (Tool Definition Class)
    
    这是一个数据类（dataclass），用于存储单个工具的所有信息。
    
    什么是 @dataclass？
    ------------------
    @dataclass 是 Python 3.7+ 引入的装饰器，它会自动为类生成：
    - __init__() 方法：根据类属性自动创建构造函数
    - __repr__() 方法：生成可读的字符串表示
    - __eq__() 方法：比较两个实例是否相等
    
    使用 @dataclass 前：
        class Tool:
            def __init__(self, name, description, parameters, function, requires_confirmation=False):
                self.name = name
                self.description = description
                self.parameters = parameters
                self.function = function
                self.requires_confirmation = requires_confirmation
    
    使用 @dataclass 后，只需声明属性即可，代码更简洁。
    
    属性说明 (Attributes)
    --------------------
    name : str
        工具的唯一标识名称，如 "read_file", "write_file"
        LLM 会使用这个名称来调用工具
    
    description : str
        工具的功能描述，告诉 LLM 这个工具能做什么
        这个描述会被包含在发送给 LLM 的 schema 中
    
    parameters : Dict[str, Any]
        工具的参数定义，使用 JSON Schema 格式
        这个格式是 OpenAI Function Calling 的标准格式
        示例：
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"}
            },
            "required": ["path"]
        }
    
    function : Callable
        实际执行工具功能的函数引用
        当工具被调用时，会执行这个函数
    
    requires_confirmation : bool
        是否需要用户确认才能执行
        对于危险操作（如写文件、执行命令），应设为 True
        默认值为 False
    """
    name: str                           # 工具名称（必需）
    description: str                    # 工具描述（必需）
    parameters: Dict[str, Any]          # 参数定义，JSON Schema 格式（必需）
    function: Callable                  # 执行函数（必需）
    requires_confirmation: bool = False # 是否需要确认（可选，默认 False）
    
    # 注意：带默认值的属性必须放在没有默认值的属性后面
    # 这是 Python 函数参数的规则：位置参数必须在关键字参数之前


# ============================================================================
# ToolRegistry 类定义 (Tool Registry Class Definition)
# ============================================================================

class ToolRegistry:
    """
    工具注册表类 (Tool Registry Class)
    
    这是一个注册表模式的实现，用于集中管理所有可用的工具。
    
    什么是注册表模式？
    ----------------
    注册表模式是一种设计模式，它提供一个全局访问点来存储和检索对象。
    在这里，我们用它来：
    1. 存储所有已注册的工具
    2. 提供统一的接口来注册、获取和执行工具
    3. 生成 LLM 需要的工具 schema
    
    主要方法：
    - register(): 装饰器，用于注册新工具
    - get_tool(): 根据名称获取工具
    - get_all_tools(): 获取所有工具
    - get_tools_schema(): 生成 OpenAI 格式的工具 schema
    - execute(): 执行指定的工具
    
    使用示例：
    ---------
    # 创建注册表
    registry = ToolRegistry()
    
    # 使用装饰器注册工具
    @registry.register(name="my_tool", description="...", parameters={...})
    def my_tool(arg1, arg2):
        return "result"
    
    # 执行工具
    result = registry.execute("my_tool", {"arg1": "value1", "arg2": "value2"})
    """
    
    def __init__(self):
        """
        构造函数 (Constructor)
        
        初始化一个空的工具字典。
        
        self._tools 的结构：
        {
            "tool_name_1": Tool(...),
            "tool_name_2": Tool(...),
            ...
        }
        
        命名约定：
        - 以单下划线 _ 开头的属性（如 _tools）表示"受保护的"属性
        - 这是一种约定，告诉其他开发者不应该直接访问这个属性
        - Python 不会强制执行这个约定，但这是良好的编程习惯
        """
        self._tools: Dict[str, Tool] = {}
        # 类型注解 Dict[str, Tool] 表示：
        # - 键是字符串（工具名称）
        # - 值是 Tool 对象
    
    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        requires_confirmation: bool = False
    ) -> Callable:
        """
        装饰器：注册工具 (Decorator: Register Tool)
        
        这是一个装饰器工厂函数（返回装饰器的函数）。
        
        什么是装饰器？
        ------------
        装饰器是 Python 的一个强大特性，它允许你在不修改函数代码的情况下，
        给函数添加额外的功能。
        
        装饰器的基本语法：
            @decorator
            def function():
                pass
        
        等价于：
            def function():
                pass
            function = decorator(function)
        
        什么是装饰器工厂？
        ----------------
        当装饰器需要接收参数时，我们需要一个"装饰器工厂"——
        一个返回装饰器的函数。
        
        @registry.register(name="...", description="...")
        def my_func():
            pass
        
        执行顺序：
        1. 调用 register(name="...", description="...") 返回 decorator 函数
        2. 调用 decorator(my_func) 返回 my_func（同时完成注册）
        
        参数说明 (Parameters)
        --------------------
        name : str
            工具的唯一名称
        
        description : str
            工具的功能描述
        
        parameters : Dict[str, Any]
            参数定义，JSON Schema 格式
            这个格式遵循 OpenAI Function Calling 的规范
        
        requires_confirmation : bool, optional
            是否需要用户确认，默认 False
        
        返回值 (Returns)
        ---------------
        Callable
            返回一个装饰器函数
        
        使用示例 (Usage Example)
        -----------------------
        @registry.register(
            name="read_file",
            description="读取文件内容",
            parameters={
                "type": "object",           # 参数是一个对象
                "properties": {             # 对象的属性定义
                    "path": {
                        "type": "string",   # path 参数是字符串类型
                        "description": "文件路径"
                    }
                },
                "required": ["path"]        # path 是必需参数
            }
        )
        def read_file(path: str) -> str:
            with open(path, 'r') as f:
                return f.read()
        """
        def decorator(func: Callable) -> Callable:
            """
            内部装饰器函数 (Inner Decorator Function)
            
            这个函数接收被装饰的函数，将其注册到工具表中，
            然后返回原函数（不做任何修改）。
            
            参数：
                func: 被装饰的函数
            
            返回：
                原函数（未修改）
            """
            # 创建 Tool 对象并存储到字典中
            self._tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters,
                function=func,                      # 存储函数引用
                requires_confirmation=requires_confirmation
            )
            # 返回原函数，这样被装饰的函数仍然可以正常调用
            return func
        
        # 返回装饰器函数
        return decorator
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        获取工具 (Get Tool)
        
        根据工具名称获取对应的 Tool 对象。
        
        参数 (Parameters)
        ----------------
        name : str
            工具名称
        
        返回值 (Returns)
        ---------------
        Optional[Tool]
            如果找到工具，返回 Tool 对象
            如果未找到，返回 None
        
        说明：
        ----
        dict.get(key) 方法：
        - 如果 key 存在，返回对应的值
        - 如果 key 不存在，返回 None（而不是抛出 KeyError）
        
        这比直接使用 dict[key] 更安全，因为不会抛出异常。
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """
        获取所有工具 (Get All Tools)
        
        返回所有已注册工具的列表。
        
        返回值 (Returns)
        ---------------
        List[Tool]
            包含所有 Tool 对象的列表
        
        说明：
        ----
        dict.values() 返回一个 dict_values 对象（视图对象）
        list() 将其转换为普通列表
        
        为什么要转换？
        - dict_values 是一个动态视图，会随字典变化而变化
        - 转换为 list 可以得到一个独立的快照
        """
        return list(self._tools.values())
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的 OpenAI 格式 schema (Get Tools Schema)
        
        生成符合 OpenAI Function Calling API 格式的工具定义列表。
        这个列表会被传递给 LLM，让 LLM 知道有哪些工具可用。
        
        返回值 (Returns)
        ---------------
        List[Dict[str, Any]]
            OpenAI 格式的工具 schema 列表
        
        返回格式示例：
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
            },
            ...
        ]
        
        OpenAI Function Calling 说明：
        ----------------------------
        OpenAI 的 Function Calling 功能允许 LLM 决定何时调用函数，
        以及使用什么参数。我们需要按照特定格式告诉 LLM 有哪些函数可用。
        
        格式要求：
        - type: 必须是 "function"
        - function.name: 函数名称
        - function.description: 函数描述（LLM 用这个来决定何时调用）
        - function.parameters: JSON Schema 格式的参数定义
        """
        schemas = []  # 存储所有工具的 schema
        
        # 遍历所有已注册的工具
        for tool in self._tools.values():
            # 为每个工具创建 OpenAI 格式的 schema
            schemas.append({
                "type": "function",     # 固定值，表示这是一个函数
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        
        return schemas
    
    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        执行工具 (Execute Tool)
        
        根据工具名称和参数执行对应的工具函数。
        
        参数 (Parameters)
        ----------------
        name : str
            要执行的工具名称
        
        arguments : Dict[str, Any]
            工具参数，以字典形式传递
            例如：{"path": "/tmp/test.txt", "content": "Hello"}
        
        返回值 (Returns)
        ---------------
        str
            工具执行结果（字符串形式）
            - 成功时返回函数的返回值（转换为字符串）
            - 失败时返回错误信息
        
        执行流程：
        --------
        1. 根据名称查找工具
        2. 如果工具不存在，返回错误信息
        3. 调用工具函数，传入参数
        4. 返回执行结果或错误信息
        
        关于 **arguments：
        ----------------
        ** 是 Python 的解包操作符（unpacking operator）
        
        tool.function(**arguments) 等价于：
        tool.function(key1=value1, key2=value2, ...)
        
        例如：
        arguments = {"path": "/tmp/test.txt", "content": "Hello"}
        tool.function(**arguments)
        # 等价于
        tool.function(path="/tmp/test.txt", content="Hello")
        
        这种方式允许我们动态地传递任意数量的关键字参数。
        """
        # 查找工具
        tool = self._tools.get(name)
        
        # 如果工具不存在，返回错误信息
        if not tool:
            return f"错误: 未找到工具 '{name}'"
        
        try:
            # 执行工具函数
            # **arguments 将字典解包为关键字参数
            result = tool.function(**arguments)
            
            # 返回结果
            # 如果结果不是 None，转换为字符串返回
            # 如果结果是 None，返回 "执行成功"
            return str(result) if result is not None else "执行成功"
        
        except Exception as e:
            # 捕获所有异常，返回错误信息
            # str(e) 获取异常的错误消息
            return f"执行错误: {str(e)}"


# ============================================================================
# 全局工具注册表实例 (Global Tool Registry Instance)
# ============================================================================

# 创建全局的工具注册表实例
# 这是单例模式的简化实现——整个应用程序共享同一个注册表
registry = ToolRegistry()

# 为什么使用全局实例？
# ------------------
# 1. 方便：所有模块都可以直接导入并使用这个注册表
# 2. 一致性：确保所有工具都注册到同一个地方
# 3. 简单：不需要复杂的依赖注入或工厂模式
#
# 使用方式：
# from agent.tools import registry
# @registry.register(...)
# def my_tool(...):
#     ...


# ============================================================================
# 内置工具定义 (Built-in Tool Definitions)
# ============================================================================
# 
# 以下是系统内置的工具，它们使用 @registry.register() 装饰器注册。
# 这些工具提供了基本的文件操作和命令执行功能。
#
# 每个工具的定义包括：
# 1. @registry.register() 装饰器：定义工具的元数据
# 2. 函数定义：实现工具的具体功能
# ============================================================================


@registry.register(
    name="read_file",                           # 工具名称
    description="读取指定路径的文件内容",         # 工具描述（LLM 会看到这个）
    parameters={                                # 参数定义（JSON Schema 格式）
        "type": "object",                       # 参数是一个对象
        "properties": {                         # 对象的属性
            "path": {                           # 属性名：path
                "type": "string",               # 类型：字符串
                "description": "要读取的文件路径"  # 描述
            }
        },
        "required": ["path"]                    # 必需的参数列表
    }
    # 注意：没有设置 requires_confirmation，默认为 False
    # 读取文件是安全操作，不需要用户确认
)
def read_file(path: str) -> str:
    """
    读取文件内容 (Read File Content)
    
    这是一个基本的文件读取工具。
    
    参数 (Parameters)
    ----------------
    path : str
        要读取的文件路径
        可以是相对路径或绝对路径
    
    返回值 (Returns)
    ---------------
    str
        文件内容，或错误信息
    
    异常处理：
    --------
    - FileNotFoundError: 文件不存在
    - 其他异常: 权限问题、编码问题等
    
    关于 with 语句：
    --------------
    with open(...) as f:
        ...
    
    这是 Python 的上下文管理器（Context Manager）语法。
    它确保文件在使用后会被正确关闭，即使发生异常也是如此。
    
    等价于：
    f = open(...)
    try:
        ...
    finally:
        f.close()
    
    关于 encoding='utf-8'：
    ---------------------
    指定文件编码为 UTF-8，这是最常用的文本编码。
    如果不指定，Python 会使用系统默认编码，可能导致乱码。
    """
    try:
        # 打开文件并读取内容
        # 'r' 表示只读模式（read）
        # encoding='utf-8' 指定编码
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()  # 读取全部内容并返回
    
    except FileNotFoundError:
        # 文件不存在时的错误处理
        return f"错误: 文件不存在 - {path}"
    
    except Exception as e:
        # 捕获其他所有异常
        # 例如：权限不足、编码错误等
        return f"读取文件错误: {str(e)}"


@registry.register(
    name="write_file",
    description="将内容写入指定路径的文件",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要写入的文件路径"
            },
            "content": {
                "type": "string",
                "description": "要写入的内容"
            }
        },
        "required": ["path", "content"]         # 两个参数都是必需的
    },
    requires_confirmation=True                  # 需要用户确认！
    # 写入文件是危险操作，可能覆盖重要文件
)
def write_file(path: str, content: str) -> str:
    """
    写入文件 (Write File)
    
    将内容写入指定的文件。如果文件不存在会创建，如果存在会覆盖。
    
    参数 (Parameters)
    ----------------
    path : str
        要写入的文件路径
    
    content : str
        要写入的内容
    
    返回值 (Returns)
    ---------------
    str
        成功消息或错误信息
    
    功能特点：
    --------
    - 自动创建不存在的目录
    - 覆盖已存在的文件
    - 使用 UTF-8 编码
    
    关于 os.makedirs()：
    ------------------
    os.makedirs(path, exist_ok=True)
    
    递归创建目录。exist_ok=True 表示如果目录已存在不会报错。
    
    例如：path = "/a/b/c/file.txt"
    os.makedirs("/a/b/c", exist_ok=True) 会创建 a、b、c 三个目录
    """
    try:
        import os  # 导入 os 模块（用于文件系统操作）
        
        # 获取文件所在的目录路径
        # os.path.dirname("/a/b/c/file.txt") 返回 "/a/b/c"
        dir_path = os.path.dirname(path)
        
        # 如果目录路径不为空，确保目录存在
        if dir_path:
            # exist_ok=True: 如果目录已存在，不会抛出异常
            os.makedirs(dir_path, exist_ok=True)
        
        # 打开文件并写入内容
        # 'w' 表示写入模式（write），会覆盖已有内容
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"成功写入文件: {path}"
    
    except Exception as e:
        return f"写入文件错误: {str(e)}"


@registry.register(
    name="list_directory",
    description="列出指定目录下的文件和文件夹",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目录路径，默认为当前目录",
                "default": "."                  # JSON Schema 中的默认值说明
            }
        },
        "required": []                          # 没有必需参数（path 有默认值）
    }
)
def list_directory(path: str = ".") -> str:
    """
    列出目录内容 (List Directory Contents)
    
    列出指定目录下的所有文件和文件夹。
    
    参数 (Parameters)
    ----------------
    path : str, optional
        目录路径，默认为当前目录 "."
    
    返回值 (Returns)
    ---------------
    str
        格式化的目录内容列表，或错误信息
    
    输出格式：
    --------
    📁 folder_name/
    📄 file_name.txt
    
    关于 os.listdir()：
    -----------------
    返回指定目录下所有文件和文件夹的名称列表（不包括 . 和 ..）
    
    关于 os.path.isdir()：
    --------------------
    判断路径是否是目录
    
    关于 sorted()：
    -------------
    对列表进行排序，返回新列表
    
    关于列表推导式：
    --------------
    [expression for item in iterable if condition]
    
    这是 Python 创建列表的简洁语法。
    """
    import os
    
    try:
        # 获取目录下所有项目的名称
        items = os.listdir(path)
        
        result = []  # 存储格式化后的结果
        
        # 遍历排序后的项目
        for item in sorted(items):
            # 构建完整路径
            full_path = os.path.join(path, item)
            
            # 判断是目录还是文件，添加相应的图标
            if os.path.isdir(full_path):
                result.append(f"📁 {item}/")    # 目录用文件夹图标
            else:
                result.append(f"📄 {item}")     # 文件用文档图标
        
        # 用换行符连接所有结果，如果为空则返回提示
        return "\n".join(result) if result else "目录为空"
    
    except FileNotFoundError:
        return f"错误: 目录不存在 - {path}"
    
    except Exception as e:
        return f"列出目录错误: {str(e)}"


@registry.register(
    name="execute_command",
    description="执行系统命令",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的命令"
            }
        },
        "required": ["command"]
    },
    requires_confirmation=True                  # 需要用户确认！
    # 执行系统命令是高危操作，必须用户确认
)
def execute_command(command: str) -> str:
    """
    执行系统命令 (Execute System Command)
    
    在系统 shell 中执行指定的命令。
    
    参数 (Parameters)
    ----------------
    command : str
        要执行的命令字符串
        例如："ls -la", "python --version", "git status"
    
    返回值 (Returns)
    ---------------
    str
        命令的输出（stdout 和 stderr），或错误信息
    
    安全警告：
    --------
    这是一个危险的操作！恶意命令可能：
    - 删除文件（如 rm -rf）
    - 安装恶意软件
    - 泄露敏感信息
    - 修改系统配置
    因此设置了 requires_confirmation=True，需要用户确认才能执行
    
    关于 subprocess 模块：
    --------------------
    subprocess 是 Python 用于创建子进程的标准库模块。
    它允许你启动新进程、连接到它们的输入/输出/错误管道，并获取返回码。
    
    subprocess.run() 参数说明：
    - command: 要执行的命令字符串
    - shell=True: 在 shell 中执行命令
      * 允许使用 shell 特性（如管道 |、重定向 >、通配符 * 等）
      * 安全风险：可能被注入恶意命令
    - capture_output=True: 捕获 stdout 和 stderr
      * 等价于 stdout=subprocess.PIPE, stderr=subprocess.PIPE
    - text=True: 以文本模式处理输出（而不是字节）
      * 自动将输出解码为字符串
    - timeout=60: 超时时间（秒）
      * 防止命令无限期运行
      * 超时会抛出 subprocess.TimeoutExpired 异常
    
    返回对象的属性：
    - result.stdout: 标准输出内容
    - result.stderr: 标准错误内容
    - result.returncode: 命令的退出码（0 通常表示成功）
    """
    import subprocess  # 导入子进程模块
    
    try:
        # 执行命令
        # subprocess.run() 会等待命令执行完成
        result = subprocess.run(
            command,            # 要执行的命令
            shell=True,         # 在 shell 中执行（支持管道、重定向等）
            capture_output=True,# 捕获输出
            text=True,          # 以文本模式处理（自动解码）
            timeout=60          # 60秒超时
        )
        
        # 构建输出字符串
        output = ""
        
        # 添加标准输出
        if result.stdout:
            output += result.stdout
        
        # 添加标准错误（如果有）
        # stderr 通常包含错误信息或警告
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        
        # 返回输出，如果为空则返回提示信息
        # strip() 移除首尾空白字符
        return output.strip() if output.strip() else "命令执行完成（无输出）"
    
    except subprocess.TimeoutExpired:
        # 命令执行超时
        return "错误: 命令执行超时（60秒）"
    
    except Exception as e:
        # 其他异常（如命令不存在等）
        return f"执行命令错误: {str(e)}"


@registry.register(
    name="search_files",
    description="在指定目录中搜索包含关键词的文件",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "搜索的目录路径"
            },
            "pattern": {
                "type": "string",
                "description": "搜索的关键词或正则表达式"
            },
            "file_extension": {
                "type": "string",
                "description": "限制搜索的文件扩展名，如 '.py'",
                "default": ""                   # 默认搜索所有文件
            }
        },
        "required": ["path", "pattern"]         # path 和 pattern 是必需的
    }
    # 注意：没有设置 requires_confirmation
    # 搜索是只读操作，不会修改文件，所以不需要确认
)
def search_files(path: str, pattern: str, file_extension: str = "") -> str:
    """
    搜索文件内容 (Search File Contents)
    
    在指定目录中递归搜索包含指定模式的文件。
    类似于 Linux 的 grep 命令的功能。
    
    参数 (Parameters)
    ----------------
    path : str
        要搜索的目录路径
    
    pattern : str
        搜索模式，支持正则表达式
        例如："TODO", "def .*test", "import.*os"
    
    file_extension : str, optional
        限制搜索的文件扩展名
        例如：".py" 只搜索 Python 文件
        默认为空字符串，表示搜索所有文件
    
    返回值 (Returns)
    ---------------
    str
        搜索结果，格式为：
        文件路径:行号: 匹配的行内容
        
        如果结果超过 50 条，会截断并提示
    
    功能特点：
    --------
    - 递归搜索子目录
    - 支持正则表达式
    - 大小写不敏感（re.IGNORECASE）
    - 自动跳过隐藏文件和目录（以 . 开头）
    - 自动跳过常见的忽略目录（node_modules, __pycache__, venv）
    - 限制结果数量（最多 50 条），防止输出过多
    
    关于 re 模块（正则表达式）：
    -------------------------
    re 是 Python 的正则表达式模块，用于模式匹配。
    
    常用函数：
    - re.compile(pattern, flags): 编译正则表达式，返回 Pattern 对象
      * 预编译可以提高多次匹配的性能
    - regex.search(string): 在字符串中搜索匹配
      * 返回 Match 对象（如果找到）或 None
    
    常用标志：
    - re.IGNORECASE (或 re.I): 忽略大小写
    - re.MULTILINE (或 re.M): 多行模式
    - re.DOTALL (或 re.S): 让 . 匹配换行符
    
    关于 os.walk()：
    --------------
    os.walk(path) 递归遍历目录树，返回一个生成器。
    
    每次迭代返回一个三元组：(root, dirs, files)
    - root: 当前目录的路径（字符串）
    - dirs: 当前目录下的子目录名列表
    - files: 当前目录下的文件名列表
    
    重要技巧：
    dirs[:] = [...] 可以原地修改 dirs 列表。
    这会影响 os.walk 的后续遍历行为——被移除的目录不会被遍历。
    
    为什么用 dirs[:] 而不是 dirs = [...]？
    - dirs[:] 是原地修改（修改同一个列表对象）
    - dirs = [...] 是创建新列表（不影响 os.walk 的行为）
    
    关于 enumerate()：
    ----------------
    enumerate(iterable, start=0) 返回枚举对象。
    
    每次迭代返回 (索引, 元素) 元组。
    
    例如：
    for i, item in enumerate(['a', 'b', 'c'], start=1):
        print(i, item)
    # 输出：
    # 1 a
    # 2 b
    # 3 c
    
    这里 start=1 表示索引从 1 开始，用于表示行号。
    """
    import os   # 文件系统操作
    import re   # 正则表达式
    
    results = []  # 存储搜索结果
    
    # 编译正则表达式
    # 预编译可以提高多次匹配的性能
    try:
        # re.IGNORECASE: 忽略大小写
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        # 如果正则表达式语法错误，返回错误信息
        return f"错误: 无效的正则表达式 - {pattern}"
    
    try:
        # os.walk() 递归遍历目录
        for root, dirs, files in os.walk(path):
            # 过滤要遍历的目录
            # dirs[:] = [...] 是原地修改列表的技巧
            # 这会影响 os.walk 的后续遍历行为
            dirs[:] = [d for d in dirs 
                      if not d.startswith('.')  # 跳过隐藏目录（以 . 开头）
                      and d not in ['node_modules', '__pycache__', 'venv']]  # 跳过常见忽略目录
            
            # 遍历当前目录下的文件
            for file in files:
                # 如果指定了扩展名，检查文件是否匹配
                if file_extension and not file.endswith(file_extension):
                    continue  # 跳过不匹配的文件
                
                # 跳过隐藏文件（以 . 开头）
                if file.startswith('.'):
                    continue
                
                # 构建完整的文件路径
                # os.path.join() 会根据操作系统使用正确的路径分隔符
                # Windows: "dir\\file.txt"
                # Linux/Mac: "dir/file.txt"
                file_path = os.path.join(root, file)
                
                try:
                    # 打开文件并逐行搜索
                    # encoding='utf-8': 使用 UTF-8 编码读取
                    # errors='ignore': 忽略编码错误
                    #   - 这样可以处理包含非 UTF-8 字符的文件
                    #   - 也可以跳过二进制文件（虽然可能读取到乱码）
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # enumerate(f, 1): 遍历文件的每一行，同时获取行号
                        # start=1 表示行号从 1 开始（而不是默认的 0）
                        # f 是文件对象，可以直接迭代，每次返回一行
                        for line_num, line in enumerate(f, 1):
                            # regex.search(line): 在当前行中搜索匹配
                            # 如果找到匹配，返回 Match 对象（真值）
                            # 如果没找到，返回 None（假值）
                            if regex.search(line):
                                # 构建结果字符串：文件路径:行号: 内容
                                # line.strip(): 移除行首尾的空白字符（包括换行符）
                                # [:100]: 只取前 100 个字符，防止单行过长
                                results.append(f"{file_path}:{line_num}: {line.strip()[:100]}")
                                
                                # 限制结果数量，防止输出过多
                                # 如果结果超过 50 条，停止搜索并返回
                                if len(results) >= 50:
                                    results.append("... (结果过多，已截断)")
                                    return "\n".join(results)
                
                except:
                    # 使用裸 except 捕获所有异常
                    # 这里故意不指定异常类型，因为可能有各种原因导致文件无法读取：
                    # - 权限不足 (PermissionError)
                    # - 文件被占用 (IOError)
                    # - 文件是二进制文件导致解码失败
                    # - 其他未知错误
                    # 我们选择静默跳过这些文件，继续搜索其他文件
                    continue
        
        # 返回所有搜索结果
        # "\n".join(results): 用换行符连接所有结果
        # 如果 results 为空列表，返回提示信息
        return "\n".join(results) if results else "未找到匹配的内容"
    
    except Exception as e:
        # 捕获外层的异常（如目录不存在、权限问题等）
        return f"搜索错误: {str(e)}"
