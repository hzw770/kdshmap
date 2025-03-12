from setuptools import setup

setup(
    name='kdshmap',
    version='0.1',
    description='Python package for calculating Keldysh dynamical maps for open quantum systems',
    author='Ziwen Huang, Rohan Rajmohan',
    author_email='zhuang@fnal.gov',
    packages=['kdshmap', 'kdshmap.core', 'kdshmap.utils', 'kdshmap.tests'],
    install_requires=[
        'cython>=3.0.12',
        'matplotlib>=3.10.1',
        'contourpy>=1.3.1',
        'numpy>=2.2.3',
        'qutip>=5.1.1',
        'scipy>=1.15.2',
        'pathos>=0.3.3'
    ]
)
