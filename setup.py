# -*- coding: utf-8 -*-


"""
Installation script for the Retro project
"""


from __future__ import absolute_import

from setuptools import setup, Extension, find_packages

import numpy as np
import versioneer


setup(
    name="i3cols",
    description="Columnar storage for and operation on IceCube data",
    author="Justin L. Lanfranchi",
    author_email="jll1062@phys.psu.edu",
    url="https://github.com/jllanfranchi/i3cols",
    license="MIT",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    python_requires=">=2.7",
    setup_requires=["pip>=1.8", "setuptools>18.5", "numpy>=1.11"],
    install_requires=["enum34", "numba>=0.45",],
    packages=find_packages(),
    include_dirs=[np.get_include()],
    package_data={},
    entry_points={'console_scripts': ['i3cols = i3cols.cli:main']},
)
