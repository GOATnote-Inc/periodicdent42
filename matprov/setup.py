"""Setup file for matprov CLI"""

from setuptools import setup, find_packages

setup(
    name="matprov",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "matprov=matprov.cli:cli",
        ],
    },
    python_requires=">=3.10",
)

