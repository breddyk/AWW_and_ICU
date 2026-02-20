import setuptools

__version__ = '1.0.0'

setuptools.setup(
    name='pgfgleam',
    version=__version__,
    author='Guillaume St-Onge',
    author_email='g.st-onge@northeastern.edu',
    description='Probability generating function approach for epidemic metapopulation models.',
    install_requires=['numpy','pandas','scipy','ray'],
    packages=setuptools.find_packages()
)
