"""
CLI 模块

提供命令行参数解析和交互式菜单。

用法:
    from cnnlib.cli import buildParser, getSettings, InteractiveCLI

    # 一行指令
    args = getSettings()
    print(args.model, args.dataset)

    # 交互菜单
    cli = InteractiveCLI()
    cli.run()
"""

from cnnlib.cli.interactive import InteractiveCLI
from cnnlib.cli.parser import buildParser, getSettings

__all__ = ["buildParser", "getSettings", "InteractiveCLI"]
