#!/usr/bin/env python3
"""
Asuka CLI Agent - 主入口
一个基于LLM的命令行智能助手
"""
import argparse
import sys

from rich.console import Console

console = Console()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Asuka CLI Agent - 基于LLM的命令行智能助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 启动交互式会话
  python main.py -p anthropic       # 使用Claude模型
  python main.py -c "列出当前目录"   # 执行单个命令
        """
    )
    
    parser.add_argument(
        "-p", "--provider",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM提供商 (默认从配置读取)"
    )
    
    parser.add_argument(
        "-c", "--command",
        type=str,
        default=None,
        help="执行单个命令后退出"
    )
    
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="禁用危险操作确认"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="最大迭代次数 (默认: 10)"
    )
    
    args = parser.parse_args()
    
    try:
        from agent import Agent
    except ImportError as e:
        console.print(f"[red]导入错误: {e}[/red]")
        console.print("[yellow]请确保已安装依赖: pip install -r requirements.txt[/yellow]")
        sys.exit(1)
    
    # 创建Agent实例
    try:
        agent = Agent(
            provider=args.provider,
            max_iterations=args.max_iterations
        )
        
        if args.no_confirm:
            agent.require_confirmation = False
        
    except Exception as e:
        console.print(f"[red]初始化Agent失败: {e}[/red]")
        console.print("[yellow]请检查API密钥配置是否正确[/yellow]")
        sys.exit(1)
    
    # 执行模式
    if args.command:
        # 单命令模式
        agent.chat(args.command)
    else:
        # 交互式模式
        agent.run_interactive()


if __name__ == "__main__":
    main()
