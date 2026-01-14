#!/usr/bin/env python3
"""
日志管理模块
提供统一的日志记录接口，支持多级别日志、文件输出、日志轮转等
"""

import os
import sys
import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from datetime import datetime

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# 颜色编码（用于控制台输出）
COLORS = {
    "DEBUG": "\033[94m",      # 蓝色
    "INFO": "\033[92m",       # 绿色
    "WARNING": "\033[93m",    # 黄色
    "ERROR": "\033[91m",      # 红色
    "CRITICAL": "\033[95m",   # 品红
    "RESET": "\033[0m"        # 重置
}


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    def format(self, record):
        # 根据日志级别添加颜色
        log_color = COLORS.get(record.levelname, COLORS["RESET"])
        message = super().format(record)
        return f"{log_color}{message}{COLORS['RESET']}"


class LoggerManager:
    """日志管理器（单例模式）"""

    _instance = None
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._default_config = {
                "level": "INFO",
                "file": None,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S",
                "max_size": 10 * 1024 * 1024,  # 10MB
                "backup_count": 5,
                "use_color": True
            }
            self._loggers = {}
            self._handlers = {}
            self._initialized = True

    def configure(self, config: Dict[str, Any]):
        """配置日志管理器"""
        self._default_config.update(config)

    def get_logger(self, name: str, **kwargs) -> logging.Logger:
        """获取或创建日志记录器"""
        if name in self._loggers:
            return self._loggers[name]

        # 合并配置
        logger_config = self._default_config.copy()
        logger_config.update(kwargs)

        # 创建logger
        logger = logging.getLogger(name)

        # 设置日志级别
        log_level = logger_config.get("level", "INFO")
        logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))

        # 移除现有的handler（避免重复）
        if logger.handlers:
            logger.handlers.clear()

        # 创建formatter
        formatter = logging.Formatter(
            fmt=logger_config.get("format"),
            datefmt=logger_config.get("date_format")
        )

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)

        # 如果使用彩色输出且不是Windows
        if logger_config.get("use_color") and sys.platform != "win32":
            colored_formatter = ColoredFormatter(
                fmt=logger_config.get("format"),
                datefmt=logger_config.get("date_format")
            )
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # 文件handler
        log_file = logger_config.get("file")
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # 创建轮转文件handler
            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=logger_config.get("max_size"),
                backupCount=logger_config.get("backup_count"),
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # 保存logger引用
        self._loggers[name] = logger
        self._handlers[name] = {
            "console": console_handler,
            "file": file_handler if log_file else None
        }

        return logger

    def set_level(self, name: str, level: str):
        """设置日志级别"""
        if name in self._loggers:
            log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
            self._loggers[name].setLevel(log_level)

    def add_file_handler(self, name: str, file_path: str):
        """为logger添加文件handler"""
        if name not in self._loggers:
            return

        logger = self._loggers[name]
        formatter = logger.handlers[0].formatter

        # 创建新的文件handler
        file_handler = RotatingFileHandler(
            filename=file_path,
            maxBytes=self._default_config["max_size"],
            backupCount=self._default_config["backup_count"],
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 更新handlers记录
        self._handlers[name]["file"] = file_handler

    def get_all_loggers(self) -> Dict[str, logging.Logger]:
        """获取所有logger"""
        return self._loggers.copy()


# 创建全局日志管理器实例
_logger_manager = LoggerManager()


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    use_color: bool = True,
    **kwargs
) -> logging.Logger:
    """
    设置并获取日志记录器

    参数:
        name: logger名称
        log_file: 日志文件路径
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 日志格式字符串
        use_color: 是否在控制台使用彩色输出

    返回:
        logging.Logger实例
    """
    config = {
        "level": log_level,
        "file": log_file,
        "use_color": use_color
    }

    if log_format:
        config["format"] = log_format

    config.update(kwargs)

    return _logger_manager.get_logger(name, **config)


def get_logger(name: str) -> logging.Logger:
    """
    获取已存在的日志记录器，如果不存在则使用默认配置创建

    参数:
        name: logger名称

    返回:
        logging.Logger实例
    """
    if name in _logger_manager._loggers:
        return _logger_manager._loggers[name]
    else:
        # 使用默认配置创建logger
        return setup_logger(name)


def configure_logging(config: Dict[str, Any]):
    """
    配置全局日志设置

    参数:
        config: 配置字典，包含日志相关配置
    """
    _logger_manager.configure(config)


class LoggerMixin:
    """为类添加logger的混入类"""

    @property
    def logger(self) -> logging.Logger:
        """获取类的logger"""
        if not hasattr(self, '_logger'):
            # 使用类名作为logger名称
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = get_logger(logger_name)
        return self._logger


def log_function_call(logger_name: str = None, level: str = "DEBUG"):
    """
    函数调用日志装饰器

    参数:
        logger_name: 使用的logger名称，默认为函数模块
        level: 日志级别
    """
    def decorator(func):
        nonlocal logger_name
        if logger_name is None:
            logger_name = f"{func.__module__}.{func.__name__}"

        log_level = LOG_LEVELS.get(level.upper(), logging.DEBUG)

        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)

            # 记录函数调用
            if logger.isEnabledFor(log_level):
                arg_str = ', '.join([str(arg) for arg in args])
                kwarg_str = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
                all_args = ', '.join(filter(None, [arg_str, kwarg_str]))

                logger.log(log_level, f"调用 {func.__name__}({all_args})")

            # 记录函数开始时间
            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)

                # 记录函数结束
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                if logger.isEnabledFor(log_level):
                    logger.log(
                        log_level, f"函数 {func.__name__} 执行完成，耗时: {duration:.3f}s")

                return result

            except Exception as e:
                # 记录异常
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.error(
                    f"函数 {func.__name__} 执行失败，耗时: {duration:.3f}s，异常: {str(e)}",
                    exc_info=True
                )
                raise

        # 保持函数元数据
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__

        return wrapper

    return decorator


class PerformanceLogger:
    """性能日志记录器"""

    def __init__(self, operation_name: str, logger_name: str = "performance"):
        """
        初始化性能日志记录器

        参数:
            operation_name: 操作名称
            logger_name: logger名称
        """
        self.operation_name = operation_name
        self.logger = get_logger(logger_name)
        self.start_time = None

    def __enter__(self):
        """进入上下文，开始计时"""
        self.start_time = datetime.now()
        self.logger.info(f"开始执行: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，记录耗时"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(
                f"完成执行: {self.operation_name}，耗时: {duration:.3f}s")
        else:
            self.logger.error(
                f"执行失败: {self.operation_name}，耗时: {duration:.3f}s，异常: {exc_val}",
                exc_info=True
            )

    def checkpoint(self, checkpoint_name: str):
        """记录检查点"""
        if self.start_time is None:
            return

        checkpoint_time = datetime.now()
        duration = (checkpoint_time - self.start_time).total_seconds()
        self.logger.debug(
            f"检查点 [{checkpoint_name}]: {self.operation_name}，当前耗时: {duration:.3f}s")


# 创建一些常用的预定义logger
def get_system_logger() -> logging.Logger:
    """获取系统logger"""
    return get_logger("system")


def get_api_logger() -> logging.Logger:
    """获取API logger"""
    return get_logger("api")


def get_model_logger() -> logging.Logger:
    """获取模型logger"""
    return get_logger("model")


def get_data_logger() -> logging.Logger:
    """获取数据logger"""
    return get_logger("data")


def get_evaluation_logger() -> logging.Logger:
    """获取评估logger"""
    return get_logger("evaluation")


# 初始化一些默认logger
def init_default_loggers():
    """初始化默认的logger"""
    # 系统核心logger
    setup_logger("system", log_level="INFO")

    # 各个模块的logger
    setup_logger("config", log_level="INFO")
    setup_logger("data", log_level="INFO")
    setup_logger("model", log_level="INFO")
    setup_logger("retrieval", log_level="INFO")
    setup_logger("rag", log_level="INFO")
    setup_logger("api", log_level="INFO")
    setup_logger("web", log_level="INFO")
    setup_logger("training", log_level="INFO")
    setup_logger("evaluation", log_level="INFO")
    setup_logger("performance", log_level="DEBUG")

    # 业务逻辑logger
    setup_logger("kb_manager", log_level="INFO")
    setup_logger("training_manager", log_level="INFO")
    setup_logger("evaluation_manager", log_level="INFO")
    setup_logger("export_manager", log_level="INFO")


# 使用示例
if __name__ == "__main__":
    # 初始化默认logger
    init_default_loggers()

    # 获取不同的logger
    system_logger = get_system_logger()
    api_logger = get_api_logger()

    # 记录不同级别的日志
    system_logger.debug("这是一个调试信息")
    system_logger.info("系统启动成功")
    system_logger.warning("这是一个警告")
    system_logger.error("这是一个错误")
    system_logger.critical("这是一个严重错误")

    # 使用性能日志记录器
    with PerformanceLogger("测试操作", "performance") as perf_log:
        import time
        time.sleep(0.1)
        perf_log.checkpoint("第一阶段完成")
        time.sleep(0.2)

    # 使用装饰器
    @log_function_call(level="INFO")
    def example_function(x, y):
        """示例函数"""
        return x + y

    example_function(1, 2)
