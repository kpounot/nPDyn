from distutils.core import setup, Extension


with open('../README.md', 'r') as f:
    description = f.read()


packagesList = [    'nPDyn',
                    'nPDyn.dataManipulation',
                    'nPDyn.dataParsers',
                    'nPDyn.dataTypes',
                    'nPDyn.dataTypes.models',
                    'nPDyn.deprecated',
                    'nPDyn.fit',
                    'nPDyn.plot']


setup(  name='nPDyn',
        version='alpha',
        description=description,
        author='Kevin Pounot',
        author_email='kpounot@hotmail.fr',
        url='github.com/kpounot/nPDyn',
        packages=packagesList   )

