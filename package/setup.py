import os, sys

from distutils.core import setup, Extension
from distutils.dist import Distribution
from Cython.Build import cythonize



filePath = os.path.abspath(__file__)
dirPath = filePath[:filePath.find('setup.py')]

   

with open(filePath[:filePath.find('package')] + 'README.rst', 'r') as f:
    description = f.read()


gsl_lib = ['gsl', 'gslcblas']

if 'win32' in sys.platform:
    #_Check for GSL on windows
    dist = Distribution()
    dist.parse_config_files()
    dist.parse_command_line()

    gsl_lib = dist.get_option_dict('build_ext')['library_dirs'][1]

    if 'gsl.lib' in os.listdir(gsl_lib) and 'gslcblas.lib' in os.listdir(gsl_lib):
            gsl_lib = ['gsl', 'gslcblas']

    else:
            gsl_lib = []



packagesList = ['nPDyn',
                'nPDyn.dataManipulation',
                'nPDyn.dataParsers',
                'nPDyn.dataTypes',
                'nPDyn.dataTypes.models',
                'nPDyn.fit',
                'nPDyn.plot',
                'nPDyn.lib']



pyabsco_ext = Extension("nPDyn.lib.pyabsco", 
                        [dirPath + "nPDyn/lib/src/absco.c", dirPath + "nPDyn/lib/pyabsco.pyx"],
                        include_dirs=[dirPath + "nPDyn/lib/src"],
                        libraries=gsl_lib)


setup(name='nPDyn',
      version='1.0',
      description=description,
      author='Kevin Pounot',
      author_email='kpounot@hotmail.fr',
      url='github.com/kpounot/nPDyn',
      packages=packagesList,
      package_dir={'nPDyn': dirPath + 'nPDyn'},
      package_data={'nPDyn': [dirPath + 'nPDyn/fit/D2O_data/*.dat']},
      ext_modules = cythonize([pyabsco_ext]))
