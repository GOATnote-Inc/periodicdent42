"""
Setup script for Autonomous R&D Intelligence Layer.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="autonomous-rd-platform",
    version="0.1.0",
    description="Autonomous R&D Intelligence Layer for physical experimentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Autonomous R&D Team",
    author_email="team@autonomousrd.ai",
    url="https://github.com/your-org/autonomous-rd",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.12",
    
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "numpy>=1.26.2",
        "scipy>=1.11.4",
        "sympy>=1.12",
        "torch>=2.1.1",
        "scikit-learn>=1.3.2",
        "gpytorch>=1.11",
        "pyscf>=2.3.0",
        "rdkit>=2023.9.1",
        "ase>=3.22.1",
        "psycopg2-binary>=2.9.9",
        "sqlalchemy>=2.0.23",
        "networkx>=3.2.1",
        "pint>=0.22",
        "structlog>=23.2.0",
        "pytest>=7.4.3",
        "pytest-asyncio>=0.21.1",
    ],
    
    extras_require={
        "dev": [
            "pytest-cov>=4.1.0",
            "mypy>=1.7.1",
            "black>=23.11.0",
            "ruff>=0.1.6",
            "ipython>=8.17.0",
            "jupyter>=1.0.0",
        ],
        "ml": [
            "botorch>=0.9.5",
            "shap>=0.43.0",
            "sentence-transformers>=2.2.2",
        ],
        "ui": [
            # Next.js is installed separately via npm
        ]
    },
    
    entry_points={
        "console_scripts": [
            "ard-bootstrap=scripts.bootstrap:main",
        ]
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
    ],
    
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)

