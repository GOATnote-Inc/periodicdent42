from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption("--cov", action="append", default=[])
    parser.addoption("--cov-report", action="append", default=[])

