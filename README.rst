===========================================================
Modelling and Inference of MICrobiomes Project (MIMIC)
===========================================================

Overview
---------

MIMIC: A Comprehensive Python Package for Simulating, Inferring, and Predicting 
Microbial Community Interactions

The study of microbial communities is vital for understanding their impact on 
environmental, health, and technological domains. The *Modelling and Inference of 
MICrobiomes Project* (MIMIC) introduces a Python package designed to advance the 
simulation, inference, and prediction of microbial community interactions and dynamics. 
Addressing the complex nature of microbial ecosystems, MIMIC integrates a suite of 
mathematical models, including previously used approaches such as *Generalized Lotka-
Volterra* (gLV), *Gaussian Processes* (GP), and *Vector Autoregression* (VAR) plus 
newly developed models for integrating multiomic data, to offer a comprehensive 
framework for analysing microbial dynamics. By leveraging Bayesian inference and 
machine learning techniques, MIMIC accurately infers the dynamics of microbial 
communities from empirical data, facilitating a deeper understanding of their complex 
biological processes, unveiling possible unknown ecological interactions, and enabling 
the design of microbial communities. Such insights could help to advance microbial 
ecology research, optimizing biotechnological applications, and contributing to 
environmental sustainability and public health strategies. MIMIC is designed for 
flexibility and ease of use, aiming to support researchers and practitioners in 
microbial ecology and microbiome research. This software package contributes to 
microbial ecology research and supports ecological predictions and applications, 
benefiting the scientific and applied microbiology communities.


Structure
-----------

The repository is organized into the following main directories:

- `AUTHORS.rst`: A list of authors and contributors to the project.
- `build/`: Contains files generated by the build process.
- `CONTRIBUTING.rst`: Guidelines for contributing to the project.
- `docs/`: Contains the project's documentation.
- `examples/`: Contains example scripts and notebooks demonstrating how to use the package.
- `HISTORY.rst`: A log of changes made in each version of the project.
- `LICENSE`: The license for the project.
- `mimic/`: The main directory for the project's source code.
- `README.rst`: The main README file for the project, providing an overview and basic usage examples.
- `requirements.txt`: A list of Python dependencies required to run the project.
- `setup.py`: The build script for the project.
- `tests/`: Contains unit tests for the project's code.

Installation
--------------

Prerequisites
^^^^^^^^^^^^^
Conda package manager is recommended due to dependencies on PyMC.

Python Packages
""""""""""""""""
The Python packages needed to run this package are listed in the requirements.txt file in the same workspace. To install them, run:

.. code-block:: bash

   pip install -r requirements.txt

Compilers
""""""""""
* g++ compiler is needed for the PyMC3 package.

.. Solvers
.. """"""""
.. * Solver 1
.. * Solver 2

Steps
^^^^^

#. Clone the repository.
#. Create a new conda environment using the `environment.yml` file.
#. Activate the new environment.
#. Install required packages using `pip install -r requirements.txt` from the root directory of the repository.
#. Install the package using `pip install -e .` from the root directory of the repository.
#. Run the code using the instructions below.
#. Deactivate the environment when finished.

Usage
-------

To get started with MIMIC, you can explore a variety of detailed examples and comprehensive documentation.

- **Documentation**: Visit our [complete documentation](https://yourdocumentationurl.com) for detailed guides, API references, and more.
- **Examples**: Check out our [Examples Directory](https://yourdocumentationurl.com/examples) which includes Jupyter notebooks demonstrating how to use MIMIC for different applications and scenarios.

The documentation is regularly updated with the latest information on usage, features, and examples to help you effectively utilize the MIMIC package in your research or applications.


Contributing
-------------

We welcome contributions to the MIMIC project. Please refer to our `Contribution Guidelines <CONTRIBUTING.rst>`_ for more information.

License
--------

This project is licensed under the `LICENSE <LICENSE>`_.

Acknowledgements
------------------

This project is based on methods proposed in `this paper <https://onlinelibrary.wiley.com/doi/full/10.1002/bies.201600188>`_.

Contact
--------

For questions or feedback, please `contact us <mailto:christopher.barnes@ucl.ac.uk>`_.

