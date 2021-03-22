from setuptools import find_packages
from setuptools import setup

setup(
  name='pyobjpextractor',
  py_modules=['pyobjpextractor'],
  version='0.1',
  description='python object proposal extractor tool',
  author='Raphael Alampay',
  author_email='raphael.alampay@gmail.com',
  url='https://github.com/ralampay/pyobjpextractor',
  packages=find_packages(),
  entry_points={
    'console_scripts': [
      'pyobjpextractor = pyobjpextractor.cli.__main__:main'
    ]
  }
)
