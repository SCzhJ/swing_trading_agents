# config_setup.py
import logging
from pathlib import Path
import sys
from typing import Optional

def get_project_root() -> Path:
    """
    动态获取项目根目录（pyproject.toml 所在目录）
    适用于：uv run 在任何子目录执行
    """
    current = Path.cwd()
    
    # 方法1：向上查找 pyproject.toml（最可靠）
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    
    # 方法2：如果没找到，raise a custom error
    raise FileNotFoundError("pyproject.toml 未找到，无法确定项目根目录")

def setup_logging(
    log_dir: Optional[str] = "logs",
    log_file: Optional[str] = "app.log",
    level: int = logging.DEBUG
) -> None:
    """
    配置日志，保存到项目根目录下的 log_dir/log_file
    """
    project_root = get_project_root()
    log_path = project_root / log_dir
    log_path.mkdir(exist_ok=True)  # 自动创建目录

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path / log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)  # 同时输出到终端
        ]
    )

    logging.info(f"日志已初始化，保存至: {log_path / log_file}")