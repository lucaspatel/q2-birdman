from setuptools import setup, find_packages

__version__ = '0.1.0'

setup(name='birdman',
      version=__version__,
      packages=find_packages(),
      include_package_data=True,
      entry_points={'console_scripts': ['birdman=birdman_cli:cli',],},
      py_modules=['birdman_cli']
      )