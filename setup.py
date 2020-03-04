#!/usr/bin/env python3
"""
Toolbox
-------
A personal toolbox.
"""
from setuptools import setup, find_packages

setup(
    name="Toolbox",
    version='1.0',
    author="Bram van Asseldonk",
    url="https://github.com/Basseldonk/Toolbox.git",
    description="Personal toolbox with code snippets of Bram van Asseldonk.",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        'torch',
        'pandas',
        ],
)
