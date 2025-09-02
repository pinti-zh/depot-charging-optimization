import contextlib
import io
import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


def get_logger(name="my_logger", level="info"):
    if level.lower() == "critical":
        log_level = logging.CRITICAL
    elif level.lower() == "error":
        log_level = logging.ERROR
    elif level.lower() == "warning":
        log_level = logging.WARNING
    elif level.lower() == "info":
        log_level = logging.INFO
    elif level.lower() == "debug":
        log_level = logging.DEBUG
    else:
        raise ValueError(f"Unknown log-level: {level}")

    theme = Theme(
        {
            "logging.level.critical": "bold reverse red",
            "logging.level.error": "bold red",
            "logging.level.warning": "bold yellow",
            "logging.level.info": "blue",
            "logging.level.debug": "dim cyan",
            "logging.level.notset": "dim",
        }
    )

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = RichHandler(console=Console(theme=theme, file=sys.stderr), markup=True, rich_tracebacks=True)
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    handler.setFormatter(formatter)
    logger.handlers = [handler]

    return logger


@contextlib.contextmanager
def log_stdout(logger, level="info"):
    # Backup the original stdout
    original_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        yield  # Code inside the `with` block runs here
    finally:
        sys.stdout = original_stdout  # Restore original stdout
        output = captured_output.getvalue()
        if output.strip():  # Avoid logging empty strings
            log_method = getattr(logger, level, logger.info)
            for line in output.strip().split("\n"):
                log_method(line)


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
