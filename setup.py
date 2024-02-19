import pathlib
from setuptools import setup, find_packages

readme = pathlib.Path('README.rst').read_text()
history = pathlib.Path('HISTORY.rst').read_text()


setup(name='MIMIC',
      version='0.1.0',
      description='Modelling and Inference of MICrobiomes Project (MIMIC) is a Python package dedicated to simulate, model, and predict microbial communities interactions and dynamics. It is designed to be a flexible and easy-to-use tool for researchers and practitioners in the field of microbial ecology and microbiome research.',
      long_description=readme + '\n\n' + history,
      include_package_data=True,
      author='Pedro Fontanarrosa',
      author_email='pfontanarrosa@gmail.com',
      license='MIT license',
      packages=find_packages(),
      test_suite='tests',
      url='https://github.com/ucl-cssb/MIMIC',
      zip_safe=False)
