from setuptools import setup

setup(
    name='vectorbt',
    version=0.2,
    description='Python library for backtesting trading strategies at scale',
    author='Oleg Polakow',
    author_email='olegpolakow@gmail.com',
    url='https://github.com/polakowo/vectorbt',
    packages=['vectorbt'],
    setup_requires=["numpy"],
    install_requires=['numpy', 'pandas', 'matplotlib', 'plotly', 'numba']
)
