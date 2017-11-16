VERSION = '0.1.1'

from setuptools import setup, find_packages

setup(name='vector-bt',
      version=VERSION,
      description='vector-bt',
      author='polakowo',
      url='https://github.com/polakowo/vector-bt',
      license='GPL v3',
      packages=find_packages(),
      install_requires=['numpy', 'pandas', 'pytz', 'poloniex', 'matplotlib'],
      python_requires='>=3')
