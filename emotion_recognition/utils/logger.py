import logging
from pathlib import Path
import sys

def setup_logger(name: str, log_dir: str = "logs"):
    """设置日志记录器"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path / f"{name}.log")
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s - %(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger