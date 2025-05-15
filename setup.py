
"""
setup.py: The setup script to make structpe pip-installable.
"""
from setuptools import setup, find_packages

setup(
    name="structpe",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.7",
    description="A multi-dataset library with generation & evaluation, plus CLI (JSON-based pipeline).",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourname/structpe",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            # This line exposes a CLI command named "structpe" 
            # which points to the 'main_cli' function inside structpe.run.
            "structpe=structpe.run:main_cli",
        ]
    },
)
