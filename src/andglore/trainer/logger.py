import logging
from datetime import datetime
from typing import Optional


class LoggerFormatter(logging.Formatter):
    def format(self, record):
        log = record.getMessage()

        if getattr(record, "show_time", False):
            log = f"[{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}] {log}"
        if getattr(record, "break_line", False):
            log = log + "\n"

        log = record.getMessage()
        if record.levelno == logging.INFO:
            log = log
        if record.levelno == logging.WARNING:
            log = f"[WARNING] {log}"
        if record.levelno >= logging.ERROR:
            log = f"[ERROR] {log}"

        if getattr(record, "print", False):
            print(log)

        return log


class Logger:
    def __init__(
        self,
        log_file: Optional[str] = None,
    ):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        formatter = LoggerFormatter()

        # console = logging.StreamHandler()
        # console.setFormatter(formatter)

        if log_file is None:
            log_file = "logs/temp.log"

        file = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file.setFormatter(formatter)

        # self.logger.addHandler(console)
        self.logger.addHandler(file)
