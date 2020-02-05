import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="octapy",
    version="0.0.1",
    author="Jason Tilley",
    author_email="jason.tilley@usm.edu",
    description="Ocean Connectivity and Tracking Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jasontilley/octapy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
