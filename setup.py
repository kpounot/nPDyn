import os, sys

import subprocess

from setuptools import setup, Extension
from distutils.dist import Distribution

try:
    from Cython.Build import cythonize
    import cython_gsl
except ImportError:
    os.system("python3 -m pip install Cython")
    os.system("python3 -m pip install CythonGSL")
    from Cython.Build import cythonize
    import cython_gsl

try:
    import versioneer
except ImportError:
    os.system("python3 -m pip install versioneer")
    import versioneer

filePath = os.path.abspath(__file__)
dirPath = os.path.dirname(filePath)


with open(dirPath + "/README.rst") as f:
    description = f.read()

packagesList = [
    "nPDyn",
    "nPDyn.dataManipulation",
    "nPDyn.dataParsers",
    "nPDyn.dataTypes",
    "nPDyn.models",
    "nPDyn.lmfit",
    "nPDyn.models.d2O_calibration",
    "nPDyn.plot",
    "nPDyn.lib",
]

try:
    pyabsco_ext = Extension(
        "nPDyn.lib.pyabsco",
        [
            dirPath + "/nPDyn/lib/src/absco.c",
            dirPath + "/nPDyn/lib/pyabsco.pyx",
        ],
        libraries=cython_gsl.get_libraries(),
        library_dirs=[cython_gsl.get_library_dir()],
        include_dirs=[
            cython_gsl.get_cython_include_dir(),
            dirPath + "/nPDyn/lib/src",
        ],
    )
    extMod = cythonize([pyabsco_ext])
except IndexError:
    extMod = None


setup(
    name="nPDyn",
    version=versioneer.get_version(),
    cmdclass={**versioneer.get_cmdclass()},
    description="Python package for analysis of neutron backscattering data",
    long_description=description,
    long_description_content_type="text/x-rst",
    platforms=["Windows", "Linux", "Mac OS X"],
    author="Kevin Pounot",
    author_email="kpounot@hotmail.fr",
    url="https://github.com/kpounot/nPDyn",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public "
        "License v3 or later (GPLv3+)",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    packages=packagesList,
    package_dir={"nPDyn": dirPath + "/nPDyn"},
    package_data={"nPDyn": [dirPath + "/nPDyn/models/d2O_calibration/*.dat"]},
    ext_modules=extMod,
    install_requires=[
        "CythonGSL",
        "cython",
        "scipy",
        "numpy",
        "matplotlib",
        "PyQt5==5.14",
        "h5py",
        "lmfit",
        "defusedxml",
    ],
)
