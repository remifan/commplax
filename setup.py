# Copyright 2021 The Commplax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install commplax"""

from setuptools import setup, find_packages

_dct = {}
with open('commplax/version.py') as f:
    exec(f.read(), _dct)
__version__ = _dct['__version__']

setup(name='commplax',
    version=__version__,
    description='differentiable DSP library for optical communication',
    author='Commplax team',
    author_email='remi.qr.fan@gmail.com',
    url='https://github.com/remifan/commplax',
    packages=find_packages(),
    install_requires=[
        'jax>=0.2.13',
        'jaxlib>=0.1.66',
        'equinox',
        'seaborn',
        'quantumrandom',
    ],
    extras_require={
        'dev': [
            'attr',
            'mock',
            'pytest',
            'parameterized',
            'ipykernel',
            'ipympl',
        ],
        'fs': [
            'zarr[jupyter]==2.9.5',
            's3fs',
            'fsspec'
        ],
        'test': [
            'pytest',
            'pytest-cov'
        ],
        'all': [
            'zarr[jupyter]==2.9.5',
            's3fs',
            'fsspec',
            'plotly',
            'tqdm'
        ]
    },
    license='Apache-2.0',
)
