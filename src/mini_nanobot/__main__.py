"""支持 ``python -m mini_nanobot`` 启动，等价于 ``mini-nanobot`` 命令。"""
from .cli import run

if __name__ == "__main__":
    run()
