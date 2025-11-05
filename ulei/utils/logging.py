"""
Logging infrastructure with structured logging support.
"""

import json
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ulei.utils.errors import ConfigurationError


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def __init__(self, include_extra: bool = True):
        """Initialize structured formatter.

        Args:
            include_extra: Whether to include extra fields from LogRecord
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log message
        """
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info for debug/error levels
        if record.levelno >= logging.ERROR or record.levelno <= logging.DEBUG:
            log_data.update(
                {
                    "filename": record.filename,
                    "lineno": record.lineno,
                    "funcName": record.funcName,
                }
            )

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {
                k: v
                for k, v in record.__dict__.items()
                if k
                not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "taskName",
                }
            }
            if extra_fields:
                log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str, ensure_ascii=False)


class ULEILoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding ULEI-specific context."""

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:  # type: ignore[override]
        """Process log message and kwargs to add context.

        Args:
            msg: Log message
            kwargs: Keyword arguments

        Returns:
            Processed message and kwargs
        """
        # Add run context if available
        extra = kwargs.setdefault("extra", {})

        # Add any context from the adapter's extra data
        if self.extra:
            extra.update(self.extra)

        return msg, kwargs


class LogManager:
    """Central logging configuration manager for ULEI."""

    def __init__(self) -> None:
        """Initialize log manager."""
        self._configured = False
        self._log_dir: Optional[Path] = None

    def configure(
        self,
        level: Union[str, int] = logging.INFO,
        format_type: str = "standard",
        output_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        enable_console: bool = True,
        logger_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Configure logging for ULEI.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_type: Format type ('standard' or 'structured')
            output_file: Optional log file path
            log_dir: Directory for log files (used if output_file not specified)
            enable_console: Whether to enable console logging
            logger_configs: Custom configurations for specific loggers
        """
        try:
            # Convert string level to int
            if isinstance(level, str):
                level = getattr(logging, level.upper())

            # Setup log directory
            if log_dir:
                self._log_dir = Path(log_dir)
                self._log_dir.mkdir(parents=True, exist_ok=True)
            elif output_file:
                self._log_dir = Path(output_file).parent
                self._log_dir.mkdir(parents=True, exist_ok=True)

            # Create logging configuration
            log_level = (
                level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO)
            )
            config = self._create_logging_config(
                level=log_level,
                format_type=format_type,
                output_file=output_file,
                enable_console=enable_console,
                logger_configs=logger_configs or {},
            )

            # Apply configuration
            logging.config.dictConfig(config)

            self._configured = True

            # Log configuration success
            logger = logging.getLogger("ulei.utils.logging")
            logger.info(
                "ULEI logging configured",
                extra={
                    "level": logging.getLevelName(level),
                    "format": format_type,
                    "console": enable_console,
                    "log_file": str(output_file) if output_file else None,
                },
            )

        except Exception as e:
            raise ConfigurationError(f"Failed to configure logging: {e}") from e

    def _create_logging_config(
        self,
        level: int,
        format_type: str,
        output_file: Optional[str],
        enable_console: bool,
        logger_configs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create logging configuration dictionary.

        Args:
            level: Logging level
            format_type: Format type
            output_file: Log file path
            enable_console: Enable console output
            logger_configs: Custom logger configurations

        Returns:
            Logging configuration dictionary
        """
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {},
            "handlers": {},
            "loggers": {},
            "root": {"level": level, "handlers": []},
        }

        # Setup formatters
        if format_type == "structured":
            config["formatters"]["structured"] = {
                "()": "ulei.utils.logging.StructuredFormatter",
                "include_extra": True,
            }
            formatter_name = "structured"
        else:
            config["formatters"]["standard"] = {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
            formatter_name = "standard"

        # Setup console handler
        if enable_console:
            config["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": formatter_name,
                "level": level,
            }
            config["root"]["handlers"].append("console")

        # Setup file handler
        if output_file:
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": output_file,
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
                "formatter": formatter_name,
                "level": level,
                "encoding": "utf-8",
            }
            config["root"]["handlers"].append("file")

        # Setup ULEI-specific loggers
        ulei_loggers = {
            "ulei": {"level": level, "propagate": True},
            "ulei.core": {"level": level, "propagate": True},
            "ulei.adapters": {"level": level, "propagate": True},
            "ulei.reporters": {"level": level, "propagate": True},
            "ulei.utils": {"level": level, "propagate": True},
        }

        # Add custom logger configurations
        ulei_loggers.update(logger_configs)

        config["loggers"] = ulei_loggers

        return config

    def get_logger(
        self, name: str, context: Optional[Dict[str, Any]] = None
    ) -> Union[logging.Logger, ULEILoggerAdapter]:
        """Get logger with optional context.

        Args:
            name: Logger name
            context: Optional context to add to log messages

        Returns:
            Logger or logger adapter with context
        """
        logger = logging.getLogger(name)

        if context:
            return ULEILoggerAdapter(logger, context)

        return logger

    def configure_from_dict(self, config: Dict[str, Any]) -> None:
        """Configure logging from configuration dictionary.

        Args:
            config: Logging configuration
        """
        level = config.get("level", "INFO")
        format_type = config.get("format", "standard")
        output_file = config.get("file")
        log_dir = config.get("dir")
        enable_console = config.get("console", True)
        logger_configs = config.get("loggers", {})

        self.configure(
            level=level,
            format_type=format_type,
            output_file=output_file,
            log_dir=log_dir,
            enable_console=enable_console,
            logger_configs=logger_configs,
        )

    def set_level(self, logger_name: str, level: Union[str, int]) -> None:
        """Set logging level for a specific logger.

        Args:
            logger_name: Name of the logger
            level: New logging level
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

    def create_run_logger(self, run_id: str) -> ULEILoggerAdapter:
        """Create logger with run context.

        Args:
            run_id: Evaluation run ID

        Returns:
            Logger adapter with run context
        """
        context = {"run_id": run_id}
        return self.get_logger("ulei.evaluation", context)  # type: ignore[return-value]

    def is_configured(self) -> bool:
        """Check if logging is configured.

        Returns:
            True if logging is configured
        """
        return self._configured

    def get_log_directory(self) -> Optional[Path]:
        """Get the configured log directory.

        Returns:
            Log directory path or None if not configured
        """
        return self._log_dir


# Global log manager instance
_global_log_manager: Optional[LogManager] = None


def get_log_manager() -> LogManager:
    """Get global log manager instance.

    Returns:
        LogManager instance
    """
    global _global_log_manager
    if _global_log_manager is None:
        _global_log_manager = LogManager()
    return _global_log_manager


def configure_logging(
    level: Union[str, int] = logging.INFO,
    format_type: str = "standard",
    output_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    enable_console: bool = True,
) -> None:
    """Configure ULEI logging with global manager.

    Args:
        level: Logging level
        format_type: Format type ('standard' or 'structured')
        output_file: Optional log file path
        log_dir: Directory for log files
        enable_console: Whether to enable console logging
    """
    manager = get_log_manager()
    manager.configure(
        level=level,
        format_type=format_type,
        output_file=output_file,
        log_dir=log_dir,
        enable_console=enable_console,
    )


def get_logger(
    name: str, context: Optional[Dict[str, Any]] = None
) -> Union[logging.Logger, ULEILoggerAdapter]:
    """Get logger using global manager.

    Args:
        name: Logger name
        context: Optional context

    Returns:
        Logger or logger adapter
    """
    manager = get_log_manager()
    return manager.get_logger(name, context)


def ensure_logging_configured(default_level: str = "INFO") -> None:
    """Ensure logging is configured with defaults if not already done.

    Args:
        default_level: Default logging level if not configured
    """
    manager = get_log_manager()
    if not manager.is_configured():
        configure_logging(level=default_level)
