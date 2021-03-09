# coding=utf-8
"""Install commplax."""

# install Jax and Jaxlib mannually

from setuptools import setup, find_packages

setup(name='commplax',
    version='1.0',
    packages=find_packages(),
    install_requires=[
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
