from setuptools import setup, find_packages

setup(
    name="CRMS2Map",
    version="0.1",
    packages=find_packages(),
    author="Jin Ikeda",
    author_email="jin.ikeda0401@gmail.com",
    description="A package for CRMS2Map data processing and analysis",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/jinikeda/CRMS2Map",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)