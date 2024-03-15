===========================================================
Modelling and Inference of MICrobiomes Project (MIMIC)
===========================================================

Overview
---------

MIMIC: A Comprehensive Python Package for Simulating, Inferring, and Predicting Microbial Community Interactions

The study of microbial communities is vital for understanding their impact on environmental, health, and technological domains. The *Modelling and Inference of MICrobiomes Project* (MIMIC) introduces a Python package designed to advance the simulation, inference, and prediction of microbial community interactions and dynamics. Addressing the complex nature of microbial ecosystems, MIMIC integrates a suite of mathematical models, including previously used approaches such as *Generalized Lotka-Volterra* (gLV), *Gaussian Processes* (GP), and *Vector Autoregression* (VAR) plus newly developed models for integrating multiomic data, to offer a comprehensive framework for analysing microbial dynamics. By leveraging Bayesian inference and machine learning techniques, MIMIC accurately infers the dynamics of microbial communities from empirical data, facilitating a deeper understanding of their complex biological processes, unveiling possible unknown ecological interactions, and enabling the design of microbial communities. Such insights could help to advance microbial ecology research, optimizing biotechnological applications, and contributing to environmental sustainability and public health strategies. MIMIC is designed for flexibility and ease of use, aiming to support researchers and practitioners in microbial ecology and microbiome research. This software package contributes to microbial ecology research and supports ecological predictions and applications, benefiting the scientific and applied microbiology communities.


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
* `run_sim_gMLVs.py`: Python script to run simulations.
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

