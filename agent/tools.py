"""
工具系统模块
定义工具的注册、管理和执行机制
"""
import json
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    requires_confirmation: bool = False  # 是否需要用户确认


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        requires_confirmation: bool = False
    ) -> Callable:
        """
        装饰器：注册工具
        
        使用方法:
            @registry.register(
                name="read_file",
                description="读取文件内容",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径"}
                    },
                    "required": ["path"]
                }
            )
            def read_file(path: str) -> str:
                ...
        """
        def decorator(func: Callable) -> Callable:
            self._tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters,
                function=func,
                requires_confirmation=requires_confirmation
            )
            return func
        return decorator
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的OpenAI格式schema
        用于传递给LLM
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return schemas
    
    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        执行工具
        
        Args:
            name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果（字符串）
        """
        tool = self._tools.get(name)
        if not tool:
            return f"错误: 未找到工具 '{name}'"
        
        try:
            result = tool.function(**arguments)
            return str(result) if result is not None else "执行成功"
        except Exception as e:
            return f"执行错误: {str(e)}"


# 全局工具注册表
registry = ToolRegistry()


# ============ 内置工具定义 ============

@registry.register(
    name="read_file",
    description="读取指定路径的文件内容",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要读取的文件路径"
            }
        },
        "required": ["path"]
    }
)
def read_file(path: str) -> str:
    """读取文件内容"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"错误: 文件不存在 - {path}"
    except Exception as e:
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
        "required": ["path", "content"]
    },
    requires_confirmation=True
)
def write_file(path: str, content: str) -> str:
    """写入文件"""
    try:
        import os
        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
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
                "default": "."
            }
        },
        "required": []
    }
)
def list_directory(path: str = ".") -> str:
    """列出目录内容"""
    import os
    try:
        items = os.listdir(path)
        result = []
        for item in sorted(items):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                result.append(f"📁 {item}/")
            else:
                result.append(f"📄 {item}")
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
    requires_confirmation=True
)
def execute_command(command: str) -> str:
    """执行系统命令"""
    import subprocess
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output.strip() if output.strip() else "命令执行完成（无输出）"
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时（60秒）"
    except Exception as e:
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
                "default": ""
            }
        },
        "required": ["path", "pattern"]
    }
)
def search_files(path: str, pattern: str, file_extension: str = "") -> str:
    """搜索文件内容"""
    import os
    import re
    
    results = []
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return f"错误: 无效的正则表达式 - {pattern}"
    
    try:
        for root, dirs, files in os.walk(path):
            # 跳过隐藏目录和常见的忽略目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
            
            for file in files:
                if file_extension and not file.endswith(file_extension):
                    continue
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(f"{file_path}:{line_num}: {line.strip()[:100]}")
                                if len(results) >= 50:  # 限制结果数量
                                    results.append("... (结果过多，已截断)")
                                    return "\n".join(results)
                except:
                    continue
        
        return "\n".join(results) if results else "未找到匹配的内容"
    except Exception as e:
        return f"搜索错误: {str(e)}"
