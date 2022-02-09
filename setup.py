import os, sys

import subprocess

from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext
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
    "nPDyn.dataParsers",
    "nPDyn.dataParsers.instruments",
    "nPDyn.models",
    "nPDyn.lmfit",
    "nPDyn.models.d2O_calibration",
    "nPDyn.plot",
    "nPDyn.lib",
]


class CommandMixin:
    user_options = [("gsl-path=", None, "path to GSL include directory.")]

    def initialize_options(self):
        super().initialize_options()
        # Initialize options
        self.gsl_path = ""

    def finalize_options(self):
        # Validate options
        if self.gsl_path != "":
            if not os.path.exists(self.gsl_path):
                raise ValueError("Path to GSL does not exist.")
        super().finalize_options()

    def run(self):
        # Use options
        global gsl_path
        gsl_path = self.gsl_path

        super().run()


class BuildExtCommand(build_ext):
    def build_extensions(self):
        for e in self.extensions:
            e.include_dirs.append(gsl_path.rstrip("/") + "/include")
            e.library_dirs.append(gsl_path.rstrip("/") + "/lib/Release")
        if "win32" in sys.platform:
            if gsl_path != "":
                build_ext.build_extensions(self)
        else:
            build_ext.build_extensions(self)


class InstallCommand(CommandMixin, install):
    user_options = (
        getattr(install, "user_options", []) + CommandMixin.user_options
    )


class DevelopCommand(CommandMixin, develop):
    user_options = (
        getattr(develop, "user_options", []) + CommandMixin.user_options
    )


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
            cython_gsl.get_include(),
            dirPath + "/nPDyn/lib/src",
        ],
        language="c++",
    )
    extMod = pyabsco_ext
except IndexError:
    extMod = None


setup(
    name="nPDyn",
    version=versioneer.get_version(),
    cmdclass={
        **versioneer.get_cmdclass(),
        "install": InstallCommand,
        "develop": DevelopCommand,
        "build_ext": BuildExtCommand,
    },
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
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    packages=packagesList,
    package_dir={"nPDyn": dirPath + "/nPDyn"},
    package_data={"nPDyn": [dirPath + "/nPDyn/models/d2O_calibration/*.dat"]},
    ext_modules=cythonize([extMod]),
    install_requires=[
        "CythonGSL",
        "cython",
        "scipy",
        "numpy",
        "matplotlib",
        "PyQt5",
        "h5py",
        "lmfit",
        "defusedxml",
        "qtwidgets",
    ],
)
