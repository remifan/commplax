# coding=utf-8
"""Install commplax."""

from setuptools import setup, find_packages

setup(name='commplax',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'jax==0.2.13',
        'jaxlib=0.1.66',
        'flax=0.3.4',
        'scipy',
        'pandas',
        'seaborn',
        'tqdm',
        'quantumrandom'
    ],
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
