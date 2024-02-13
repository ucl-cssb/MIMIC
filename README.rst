===========================================================
Modelling and Inference of MICrobiomes Project (MIMIC)
===========================================================

Overview
---------

The MIMIC project utilizes generalized Lotka-Volterra (gLV) models to predict the dynamics of microbial communities. This repository contains Python code developed for modeling, simulating, and analyzing these dynamics.

Structure
-----------

The repository is organized into the following main directories:

* `.github/`: Contains templates and workflows for GitHub features.

  * `ISSUE_TEMPLATE/`: Directory for issue templates.
  * `PULL_REQUEST_TEMPLATE.md`: Markdown file for pull request template.
  * `workflows/`: Directory for GitHub Actions workflows.

* `.gitignore`: Specifies intentionally untracked files to ignore.
* `.vscode/`: Contains configuration files for Visual Studio Code.

  * `launch.json`: Configures debugger settings.
  * `settings.json`: Specifies VS Code settings.

* `CONTRIBUTING.rst`: Guidelines for contributing to the project.
* `data/`: Contains data files and scripts for the project.

  * CSV files: Data files for the project.
  * `process.R`: R script for processing data.

* `data_analysis/`: Contains Python scripts for data analysis.

  * `load_data.py`: Python script for loading data.

* `examples/`: Contains Jupyter notebooks with examples.

  * `*.ipynb`: Jupyter notebooks.

* `gMLV/`: Contains Python scripts for the project and core code for the gLV model.

  * `cLV.py`: Python script.

* `README.rst`: Provides an overview of the project.
* `requirements.txt`: Lists Python dependencies for the project.
* `run_gMLV_sims.py`: Python script to run simulations.
* `setup.py`: Python script for setting up the project.

Installation
--------------

Prerequisites
^^^^^^^^^^^^^

* Python 3.10
* Conda package manager

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

Running gLV Model Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python3 run_gLV.py <output directory> <number to simulate> <number of species> <number of time points> <number of replicates> <number of time points to fit> <number of replicates to fit> <number of time points to predict> <number of replicates to predict>

Example:

.. code-block:: bash

    python run_gLV.py "C:\Users\User\Desktop\test_gLV" 100 10 100 10 50 5 50 5

Generating gLV Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python3 run_gLV_sims.py <output directory> <number to simulate>

Example:

.. code-block:: bash

    python run_gLV_sims.py "C:\Users\User\Desktop\test_gLV" 100

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

For questions or feedback, please `contact us <mailto:contact@example.com>`_.

