# This file provides a common logger for all modules in the project

import logging  # python built-in logging module
import os  # access env variables

def get_logger(name: str) -> logging.Logger:
    # create logger with given module name
    logger = logging.getLogger(name)

    # set log level to INFO — logs INFO, WARNING, ERROR, CRITICAL
    logger.setLevel(logging.INFO)

    # avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # create console handler — logs to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # handler level INFO

    # create logs directory if not exists
    os.makedirs('./logs', exist_ok=True)

    # create file handler — logs to file
    file_handler = logging.FileHandler('./logs/app.log')
    file_handler.setLevel(logging.INFO)  # handler level INFO

    # define log format — time, module name, level, message
    formatter = logging.Formatter(
        '%(asctime)s — %(name)s — %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler.setFormatter(formatter)  # apply format to console handler
    file_handler.setFormatter(formatter)     # apply format to file handler

    logger.addHandler(console_handler)  # add console handler to logger
    logger.addHandler(file_handler)     # add file handler to logger

    return logger  # return configured loggers