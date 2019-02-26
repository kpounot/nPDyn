from distutils.core import setup, Extension


with open('README.md', 'r') as f:
    description = f.read()


packagesList = [    'package',
                    'package.dataManipulation',
                    'package.dataParsers',
                    'package.dataTypes',
                    'package.dataTypes.models',
                    'package.deprecated',
                    'package.fit',
                    'package.plot',
                    'package.NAMDAnalyzer']

setup(  name='nPDyn',
        version='alpha',
        description=description,
        author='Kevin Pounot',
        author_email='kpounot@hotmail.fr',
        url='github.com/kpounot/nPDyn',
        packages=packagesList   )

