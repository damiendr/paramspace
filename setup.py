#!/usr/bin/env python
from setuptools import setup, find_packages # Always prefer setuptools over distutils
from codecs import open # To use a consistent encoding
from os import path


# Get the long description from the relevant file:
here = path.abspath(path.dirname(__file__))
readme = path.join(here, 'README.md')
try:
    from pypandoc import convert
    long_description = convert(readme, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    with open(readme, 'r', encoding='utf-8') as f:
        long_description = f.read()


setup(
    name='paramspace',
    description='Generating parameter spaces from model definitions',
    long_description=long_description,
    version='0.1',
    author='Damien Drix',
    author_email='damien.drix@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    packages=['paramspace'],
    install_requires=[
        "hyperopt>=0.0.2",
        "scipy>=0.13.3",
        "traits>=4.5.0"
    ],
)
