from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

version = {}
with open("vectorbt/_version.py") as fp:
    exec(fp.read(), version)

setup(
    name='vectorbt',
    version=version['__version__'],
    description='Python library for backtesting and analyzing trading strategies at scale',
    author='Oleg Polakow',
    author_email='olegpolakow@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/polakowo/vectorbt',
    packages=find_packages(),
    package_data={
        'vectorbt': ['templates/*.json']
    },
    install_requires=[
        'numpy>=1.16.5',
        'pandas',
        'scipy',
        'matplotlib',
        'plotly>=4.12.0',
        'ipywidgets>=7.0.0',
        'numba>=0.53.1',
        'dill',
        'tqdm',
        'dateparser',
        'imageio',
        'scikit-learn',
        'schedule',
        'requests',
        'pytz',
        'typing_extensions; python_version < "3.8"',
        'mypy_extensions',
        'kaleido'
    ],
    extras_require={
        'full': [
            'yfinance',
            'python-binance',
            'ccxt',
            'ray; python_version < "3.9"',
            'empyrical',
            'ta',
            'pandas_ta',
            'TA-Lib',
            'python-telegram-bot>=13.4'  # LGPLv3
        ],
        'cov': [
            'pytest',
            'pytest-cov',
            'codecov'
        ],
        'docs': [
            'pdoc3'  # AGPL-3.0
        ]
    },
    python_requires='>=3.6, <3.10',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
)
