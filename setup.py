from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vectorbt',
    version='0.15.2',
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
    license='GPLv3+',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Software Development',
        'Topic :: Office/Business :: Financial',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent'
    ],
)
