# coding=utf-8
"""Install commplax."""

from setuptools import setup, find_packages

setup(name='commplax',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'jax',
        'scipy',
        'pandas',
        'seaborn',
        'tqdm',
        'quantumrandom'
    ],
    extras_requre={
        'cpu': [
            'jaxlib'
        ],
    },
    extras_require={
        'dev': [
            'attr',
            'mock',
            'pytest',
            'parameterized',
            'h5py',
            'jupyter',
            'ipykernel',
            'ipympl'
        ],
    },
)
