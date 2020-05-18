from setuptools import setup, find_packages

setup(
    name='vectorbt',
    version=0.7,
    description='Python library for backtesting trading strategies at scale',
    author='Oleg Polakow',
    author_email='olegpolakow@gmail.com',
    url='https://github.com/polakowo/vectorbt',
    packages=find_packages(),
    install_requires=[
        'numpy', 
        'pandas', 
        'matplotlib', 
        'ipywidgets', 
        'plotly', 
        'numba>=0.49.0', 
        'scipy'
    ]
)