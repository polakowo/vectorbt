from setuptools import setup
from vectorbt import __version__

setup(
    name='vectorbt',
    version=__version__,
    description='A tiny, fully vectorized Python library for backtesting trading strategies at scale',
    author='Oleg Polakow',
    author_email='olegpolakow@gmail.com',
    url='https://github.com/polakowo/vectorbt',
    packages=['vectorbt'],
    install_requires=['numpy', 'pandas', 'matplotlib', 'plotly', 'numba']
)