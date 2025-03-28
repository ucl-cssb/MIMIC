import pathlib
from setuptools import setup, find_packages

# Get the directory where setup.py is located
here = pathlib.Path(__file__).parent

readme = (here / 'README.rst').read_text()
history = (here / 'HISTORY.rst').read_text()

# Read the requirements from requirements.txt
with open(here / 'requirements.in') as f:
    requirements = f.read().splitlines()

setup(
    name='MIMIC',
    version='0.1.0',
    description='Modelling and Inference of MICrobiomes Project (MIMIC) is a Python package dedicated to simulate, model, and predict microbial communities interactions and dynamics. It is designed to be a flexible and easy-to-use tool for researchers and practitioners in the field of microbial ecology and microbiome research.',
    long_description=readme +
    '\n\n' +
    history,
    include_package_data=True,
    author='Pedro Fontanarrosa',
    author_email='pfontanarrosa@gmail.com',
    license='MIT license',
    packages=find_packages(include=['mimic', 'mimic.*']),
    test_suite='tests',
    url='https://github.com/ucl-cssb/MIMIC',
    zip_safe=False,
    install_requires=requirements,
)
