import logging

import pytest

from nonconform._internal.log_utils import get_logger


@pytest.mark.parametrize(
    "name", ["test_module", "module.submodule", "test_module_name", "", "a" * 200]
)
def test_logger_is_namespaced_and_reused(name):
    logger = get_logger(name)

    assert isinstance(logger, logging.Logger)
    assert logger.name == f"nonconform.{name}"
    assert get_logger(name) is logger


def test_package_root_logger_is_configured_once(monkeypatch):
    root_logger = logging.getLogger("nonconform")
    original_level = root_logger.level
    monkeypatch.setattr(root_logger, "handlers", [])
    monkeypatch.setattr(root_logger, "propagate", True)
    root_logger.setLevel(logging.NOTSET)
    try:
        get_logger("first")
        get_logger("second")
        level = root_logger.level
    finally:
        root_logger.setLevel(original_level)

    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0], logging.StreamHandler)
    assert level == logging.INFO
    assert root_logger.propagate is False


def test_default_level_emits_info_and_above():
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = get_logger("test_levels")
    logger.addHandler(handler)
    try:
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
    finally:
        logger.removeHandler(handler)

    assert [(record.levelno, record.getMessage()) for record in records] == [
        (logging.INFO, "info message"),
        (logging.WARNING, "warning message"),
        (logging.ERROR, "error message"),
    ]
