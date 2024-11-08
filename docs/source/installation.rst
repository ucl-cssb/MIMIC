.. highlight:: shell

============
Installation
============

Prerequisites
^^^^^^^^^^^^^

- **Conda Package Manager**: We recommend using Conda to manage the environment due to dependencies that may not be available via pip.

Installation Steps
^^^^^^^^^^^^^^^^^^^

For macOS and Ubuntu
""""""""""""""""""""

1. **Clone the Repository**

   .. code-block:: bash

      git clone https://github.com/ucl-cssb/MIMIC.git
      cd MIMIC

2. **Create the Conda Environment**

   .. code-block:: bash

      conda env create -f environment.yml

3. **Activate the Environment**

   .. code-block:: bash

      conda activate mimic_env

4. **Install the Package**

   .. code-block:: bash

      pip install -e .

5. **Run the Code**

   Refer to the `Usage`_ section below for instructions on how to run the code.

For Windows
"""""""""""

1. **Clone the Repository**

   .. code-block:: bash

      git clone https://github.com/ucl-cssb/MIMIC.git
      cd MIMIC

2. **Create the Conda Environment for Windows**

   On Windows, use the `environment_windows.yml` file:

   .. code-block:: bash

      conda env create -f environment_windows.yml

3. **Activate the Environment**

   .. code-block:: bash

      conda activate mimic_env

4. **Install the Package**

   Install the package in editable mode:

   .. code-block:: bash

      pip install -e .

5. **Run the Code**

   Refer to the `Usage`_ section below for instructions on how to run the code.

Alternative Installation Using Pip Only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to use pip without Conda, you can install the package and its dependencies by compiling `requirements.in` into `requirements.txt`:

.. code-block:: bash

   # Step 1: Compile requirements.txt from requirements.in
   pip install pip-tools
   pip-compile requirements.in

   # Step 2: Install dependencies
   pip install -r requirements.txt
   pip install -e .

**Note**: This method may not install all dependencies correctly, especially if there are packages that are only available via Conda. We recommend using the Conda installation method for full functionality.

Compilers
""""""""""
A g++ compiler is required for the PyMC3 package.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/