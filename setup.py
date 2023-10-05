from setuptools import setup

setup(
    name='kdshmap',
    version='0.1',
    description='Python package for calculating Keldysh dynamical maps for open quantum systems',
    author='Ziwen Huang',
    author_email='zhuang@fnal.gov',
    packages=['kdshmap', 'kdshmap.core', 'kdshmap.utils', 'kdshmap.tests'],
    install_requires=[
        'cython>=0.29.20,<3.0.0',
        'matplotlib>=3.5',
        'numpy>=1.14.2',
        'qutip>=4.3.1',
        'scipy>=1.1.0',
        'pathos>=0.3.1'
    ]
)
