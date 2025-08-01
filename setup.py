"""
Setup configuration for context-store package
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="context-store",
    version="0.1.0",
    author="Agent Context Template Team",
    description="Core validators and utilities for the context system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/agent-context-template",
    packages=find_packages(include=["src", "src.*"]),
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "click>=8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "validate-config=src.validators.config_validator:main",
        ],
    },
)
