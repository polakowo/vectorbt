from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vectorbt',
    version='0.13.5',
    description='Python library for backtesting and analyzing trading strategies at scale',
    author='Oleg Polakow',
    author_email='olegpolakow@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/polakowo/vectorbt',
    packages=find_packages(),
    install_requires=[
        'numpy', 
        'pandas', 
        'matplotlib', 
        'ipywidgets', 
        'plotly', 
        'numba>=0.50.1', 
        'scipy'
    ],
    python_requires='>=3.6'
)
