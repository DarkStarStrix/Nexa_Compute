from setuptools import setup, find_packages

setup(
    name="nexa-forge",
    version="0.1.0",
    description="Python SDK for Nexa Forge",
    author="Nexa AI",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
    ],
)
