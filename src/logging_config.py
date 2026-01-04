"""
Logging 配置模块

功能：
- 支持控制台输出（仅主进程）
- 支持文件输出（所有进程）
- 自动检测 DDP 环境，只有主进程输出到控制台
- 避免每次 print 都要判断是否是主进程

使用方法：

方法1：使用标准 logging 模块
    from src.logging_config import setup_logging
    import logging
    
    # 在程序开始时初始化（通常在 main 函数开始处）
    setup_logging()
    
    # 之后直接使用 logging，不需要判断是否是主进程
    logger = logging.getLogger(__name__)
    logger.info("这条消息只会从主进程输出到控制台，但所有进程都会写入文件")
    logger.warning("警告信息")
    logger.error("错误信息")

方法2：使用便捷函数
    from src.logging_config import setup_logging, info, warning, error
    
    setup_logging()
    info("信息消息")
    warning("警告消息")
    error("错误消息")

方法3：在 DDP 环境中使用（推荐）
    from src.logging_config import setup_logging
    import logging
    
    # 在 setup_ddp() 之后调用
    rank, world_size, local_rank, device = setup_ddp()
    setup_logging(rank=rank)  # 可以显式传入 rank
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using {world_size} GPU(s)")

注意事项：
- 日志文件保存在 Config.cur_run_dir 目录下
- 日志文件会自动轮转（最大 10MB，保留 5 个备份）
- 所有进程的日志都会写入文件，但只有主进程（rank 0）会输出到控制台
- 日志格式包含时间戳、日志级别、rank 信息等
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def get_rank():
    """
    获取当前进程的 rank
    
    Returns:
        int: 当前进程的 rank，如果不是 DDP 环境则返回 0
    """
    try:
        if 'RANK' in os.environ:
            return int(os.environ['RANK'])
        # 也检查是否已经初始化了分布式环境
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except (ImportError, ValueError):
        pass
    return 0


def is_main_process(rank=None):
    """
    判断是否为主进程
    
    Args:
        rank: 进程 rank，如果为 None 则自动检测
        
    Returns:
        bool: 是否为主进程
    """
    if rank is None:
        rank = get_rank()
    return rank == 0


def setup_logging(log_dir=None, log_level=logging.INFO, log_file='training.log', rank=None):
    """
    配置 logging 模块
    
    Args:
        log_dir: 日志文件保存目录，如果为 None 则使用 Config.cur_run_dir
        log_level: 日志级别，默认为 INFO
        log_file: 日志文件名，默认为 'training.log'
        rank: 当前进程的 rank，如果为 None 则自动检测
        
    Returns:
        logging.Logger: 配置好的 logger
    """
    # 获取 rank
    if rank is None:
        rank = get_rank()
    
    # 获取日志目录
    if log_dir is None:
        try:
            from .config import Config
            log_dir = Config.cur_run_dir
        except ImportError:
            # 如果无法导入 Config，使用当前目录
            log_dir = os.getcwd()
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除已有的处理器（避免重复添加）
    logger.handlers.clear()
    
    # 日志格式
    detailed_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [Rank %(rank)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加 rank 信息到日志记录
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank
        
        def filter(self, record):
            record.rank = self.rank
            return True
    
    rank_filter = RankFilter(rank)
    
    # 1. 文件处理器（所有进程都写入文件）
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_format)
    file_handler.addFilter(rank_filter)
    logger.addHandler(file_handler)
    
    # 2. 控制台处理器（只有主进程输出到控制台）
    if is_main_process(rank):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_format)
        console_handler.addFilter(rank_filter)
        logger.addHandler(console_handler)
    else:
        # 非主进程：设置控制台输出级别为 ERROR 或更高，避免输出到控制台
        # 但为了安全，我们直接不添加控制台处理器
        pass
    
    return logger


def get_logger(name=None):
    """
    获取 logger 实例
    
    Args:
        name: logger 名称，如果为 None 则返回 root logger
        
    Returns:
        logging.Logger: logger 实例
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)


# 便捷函数：直接使用 logging 模块的标准方法
def info(message, *args, **kwargs):
    """记录 INFO 级别日志"""
    logging.info(message, *args, **kwargs)


def warning(message, *args, **kwargs):
    """记录 WARNING 级别日志"""
    logging.warning(message, *args, **kwargs)


def error(message, *args, **kwargs):
    """记录 ERROR 级别日志"""
    logging.error(message, *args, **kwargs)


def debug(message, *args, **kwargs):
    """记录 DEBUG 级别日志"""
    logging.debug(message, *args, **kwargs)


def critical(message, *args, **kwargs):
    """记录 CRITICAL 级别日志"""
    logging.critical(message, *args, **kwargs)
