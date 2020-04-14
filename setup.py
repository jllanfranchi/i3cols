# -*- coding: utf-8 -*-


"""
Installation script for the Retro project
"""


from __future__ import absolute_import

import sys

from setuptools import setup, find_packages

#import versioneer


with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["numba>=0.45"]
if sys.version_info < (3, 4):
    install_requires.append("enum34")

setup(
    name="i3cols",
    version="0.1.3",
    author="Justin L. Lanfranchi",
    author_email="jll1062@phys.psu.edu",
    description="Numpy columnar storage for IceCube data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jllanfranchi/i3cols",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    #version=versioneer.get_version(),
    #cmdclass=versioneer.get_cmdclass(),
    python_requires=">=2.7",
    setup_requires=["pip>=1.8", "setuptools>18.5"],
    install_requires=install_requires,
    package_data={},
    entry_points={'console_scripts': ['i3cols = i3cols.cli:main']},
)
