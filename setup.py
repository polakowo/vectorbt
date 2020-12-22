from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vectorbt',
    version='0.15.3',
    description='Python library for backtesting and analyzing trading strategies at scale',
    author='Oleg Polakow',
    author_email='olegpolakow@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/polakowo/vectorbt',
    packages=find_packages(),
    package_data={
        'vectorbt': ['templates/*.json']
    },
    install_requires=[
        'numpy', 
        'pandas',
        'scipy',
        'matplotlib',
        'plotly>=4.12.0',
        'ipywidgets>=7.0.0',
        'numba>=0.51.2'
    ],
    python_requires='>=3.6, <3.9',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
)
