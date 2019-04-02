import os

from distutils.core import setup, Extension
from Cython.Build import cythonize



with open('../README.md', 'r') as f:
    description = f.read()



packagesList = [    'nPDyn',
                    'nPDyn.dataManipulation',
                    'nPDyn.dataParsers',
                    'nPDyn.dataTypes',
                    'nPDyn.dataTypes.models',
                    'nPDyn.deprecated',
                    'nPDyn.fit',
                    'nPDyn.plot',
                    'nPDyn.lib'     ]



pyabsco_ext = Extension( "nPDyn.lib.pyabsco", 
                         ["nPDyn/lib/src/absco.c", "nPDyn/lib/pyabsco.pyx"],
                         include_dirs=["nPDyn/lib/src"],
                         libraries=["gsl"] )


setup(  name='nPDyn',
        version='alpha',
        description=description,
        author='Kevin Pounot',
        author_email='kpounot@hotmail.fr',
        url='github.com/kpounot/nPDyn',
        packages=packagesList,
        ext_modules = cythonize([pyabsco_ext]) )

