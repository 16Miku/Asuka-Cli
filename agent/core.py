"""
Agent核心模块
实现Agent的主循环逻辑
"""
import json
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .llm import LLMClient, Message
from .tools import registry

console = Console()


class Agent:
    """CLI Agent核心类"""
    
    SYSTEM_PROMPT = """你是一个强大的CLI助手，可以帮助用户完成各种任务。

你可以使用以下工具来完成任务：
- read_file: 读取文件内容
- write_file: 写入文件内容
- list_directory: 列出目录内容
- execute_command: 执行系统命令
- search_files: 搜索文件内容

使用工具时，请：
1. 先分析用户的需求
2. 选择合适的工具
3. 执行工具并观察结果
4. 根据结果决定下一步行动

如果任务完成，请直接回复用户结果。
如果需要更多信息，请询问用户。
"""
    
    def __init__(
        self,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    ):
        """
        初始化Agent
        
        Args:
            provider: LLM提供商 (openai/anthropic)
            system_prompt: 自定义系统提示词
            max_iterations: 最大迭代次数（防止无限循环）
        """
        self.llm = LLMClient(provider)
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.max_iterations = max_iterations
        self.messages: List[Message] = []
        self.require_confirmation = True  # 是否需要用户确认危险操作
        
        # 初始化系统消息
        self._init_messages()
    
    def _init_messages(self):
        """初始化消息历史"""
        self.messages = [
            Message(role="system", content=self.system_prompt)
        ]
    
    def reset(self):
        """重置对话历史"""
        self._init_messages()
        console.print("[dim]对话已重置[/dim]")
    
    def _confirm_action(self, tool_name: str, arguments: Dict) -> bool:
        """请求用户确认危险操作"""
        tool = registry.get_tool(tool_name)
        if not tool or not tool.requires_confirmation:
            return True
        
        if not self.require_confirmation:
            return True
        
        console.print(Panel(
            f"[yellow]工具:[/yellow] {tool_name}\n"
            f"[yellow]参数:[/yellow] {json.dumps(arguments, ensure_ascii=False, indent=2)}",
            title="⚠️ 需要确认",
            border_style="yellow"
        ))
        
        response = console.input("[yellow]是否执行此操作? (y/n): [/yellow]").strip().lower()
        return response in ['y', 'yes', '是']
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Message]:
        """执行工具调用并返回结果消息"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}
            
            # 显示工具调用
            console.print(f"\n[cyan]🔧 调用工具:[/cyan] {tool_name}")
            console.print(f"[dim]参数: {json.dumps(arguments, ensure_ascii=False)}[/dim]")
            
            # 确认危险操作
            if not self._confirm_action(tool_name, arguments):
                result = "用户取消了此操作"
                console.print(f"[yellow]⏹️ 操作已取消[/yellow]")
            else:
                # 执行工具
                result = registry.execute(tool_name, arguments)
                
                # 显示结果（截断过长的输出）
                display_result = result[:500] + "..." if len(result) > 500 else result
                console.print(f"[green]📤 结果:[/green]\n{display_result}")
            
            # 添加工具结果消息
            results.append(Message(
                role="tool",
                content=result,
                tool_call_id=tool_call["id"]
            ))
        
        return results
    
    def chat(self, user_input: str) -> str:
        """
        处理用户输入并返回响应
        
        Args:
            user_input: 用户输入
            
        Returns:
            Agent的最终响应
        """
        # 添加用户消息
        self.messages.append(Message(role="user", content=user_input))
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            # 调用LLM
            console.print(f"\n[dim]思考中... (迭代 {iteration}/{self.max_iterations})[/dim]")
            
            try:
                response = self.llm.chat(
                    messages=self.messages,
                    tools=registry.get_tools_schema()
                )
            except Exception as e:
                error_msg = f"LLM调用错误: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                return error_msg
            
            # 处理响应
            content = response.get("content", "")
            tool_calls = response.get("tool_calls")
            
            # 如果有文本内容，显示出来
            if content:
                console.print(Panel(
                    Markdown(content),
                    title="🤖 Assistant",
                    border_style="blue"
                ))
            
            # 添加助手消息
            self.messages.append(Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls
            ))
            
            # 如果没有工具调用，说明任务完成
            if not tool_calls:
                return content
            
            # 执行工具调用
            tool_results = self._execute_tool_calls(tool_calls)
            self.messages.extend(tool_results)
        
        # 达到最大迭代次数
        warning = f"⚠️ 达到最大迭代次数 ({self.max_iterations})，停止执行"
        console.print(f"[yellow]{warning}[/yellow]")
        return warning
    
    def run_interactive(self):
        """运行交互式会话"""
        console.print(Panel(
            "[bold green]Asuka CLI Agent[/bold green]\n"
            "输入你的问题或任务，我会帮你完成。\n"
            "输入 [cyan]/help[/cyan] 查看帮助，[cyan]/quit[/cyan] 退出。",
            title="欢迎",
            border_style="green"
        ))
        
        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        continue
                    else:
                        break
                
                # 处理用户输入
                self.chat(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]使用 /quit 退出[/yellow]")
            except EOFError:
                break
        
        console.print("\n[dim]再见！[/dim]")
    
    def _handle_command(self, command: str) -> bool:
        """
        处理斜杠命令
        
        Returns:
            True 继续运行，False 退出
        """
        cmd = command.lower().strip()
        
        if cmd in ["/quit", "/exit", "/q"]:
            return False
        
        elif cmd in ["/help", "/h", "/?"]:
            console.print(Panel(
                "[cyan]/help[/cyan]  - 显示帮助\n"
                "[cyan]/reset[/cyan] - 重置对话\n"
                "[cyan]/tools[/cyan] - 显示可用工具\n"
                "[cyan]/auto[/cyan]  - 切换自动确认模式\n"
                "[cyan]/quit[/cyan]  - 退出程序",
                title="帮助",
                border_style="cyan"
            ))
        
        elif cmd == "/reset":
            self.reset()
        
        elif cmd == "/tools":
            tools = registry.get_all_tools()
            tools_info = "\n".join([
                f"[cyan]{t.name}[/cyan]: {t.description}"
                for t in tools
            ])
            console.print(Panel(tools_info, title="可用工具", border_style="cyan"))
        
        elif cmd == "/auto":
            self.require_confirmation = not self.require_confirmation
            status = "关闭" if self.require_confirmation else "开启"
            console.print(f"[yellow]自动确认模式已{status}[/yellow]")
        
        else:
            console.print(f"[red]未知命令: {command}[/red]")
        
        return True
